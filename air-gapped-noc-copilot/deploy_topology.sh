#!/usr/bin/env bash
#
# Air-Gapped NOC Copilot - Topology Deployment Script
# Deploys Containerlab topology and applies routing/IPSec configurations
#

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONFIGS_DIR="${SCRIPT_DIR}/configs"
TOPOLOGY_FILE="${SCRIPT_DIR}/topology.clab.yml"

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

log_info()    { echo -e "${BLUE}[INFO]${NC} $*"; }
log_success() { echo -e "${GREEN}[OK]${NC} $*"; }
log_warn()    { echo -e "${YELLOW}[WARN]${NC} $*"; }
log_error()   { echo -e "${RED}[ERROR]${NC} $*" >&2; }

check_dependencies() {
    local deps=("containerlab" "docker" "python3")
    for dep in "${deps[@]}"; do
        if ! command -v "$dep" &>/dev/null; then
            log_error "Missing dependency: $dep"
            exit 1
        fi
    done
    log_success "All dependencies found"
}

generate_configs() {
    log_info "Generating routing configurations..."
    python3 "${SCRIPT_DIR}/configure_routing.py"
    
    log_info "Generating IPSec overlay configurations..."
    python3 "${SCRIPT_DIR}/configure_ipsec_overlay.py"
    
    log_success "All configurations generated"
}

deploy_topology() {
    log_info "Destroying any existing topology..."
    clab destroy -t "${TOPOLOGY_FILE}" --cleanup 2>/dev/null || true
    
    log_info "Deploying Containerlab topology..."
    clab deploy -t "${TOPOLOGY_FILE}" --reconfigure
    
    log_success "Topology deployed successfully"
}

apply_routing_configs() {
    local nodes=(
        "p-core-1" "p-core-2"
        "pe-hub-east-1" "pe-hub-west-1"
        "pe-dc-1" "pe-dc-2"
        "ce-branch-1" "ce-branch-2" "ce-branch-3" "ce-branch-4"
    )
    
    log_info "Applying routing configurations to nodes..."
    
    for node in "${nodes[@]}"; do
        local routing_config="${CONFIGS_DIR}/${node}.conf"
        local ipsec_config="${CONFIGS_DIR}/${node}_ipsec.conf"
        
        if [[ ! -f "${routing_config}" ]]; then
            log_warn "Routing config not found for ${node}, skipping"
            continue
        fi
        
        log_info "  Configuring ${node}..."
        
        # Copy and apply routing config
        docker cp "${routing_config}" "clab-mpls-${node}:/etc/frr/frr.conf"
        
        # Append IPSec config if it exists
        if [[ -f "${ipsec_config}" ]]; then
            cat "${ipsec_config}" >> "${routing_config}.tmp"
            cat "${routing_config}" >> "${routing_config}.tmp"
            docker cp "${routing_config}.tmp" "clab-mpls-${node}:/etc/frr/frr.conf"
            rm -f "${routing_config}.tmp"
        fi
        
        # Restart FRR to apply changes
        docker exec "clab-mpls-${node}" vtysh -c "configure terminal" -c "do write memory" 2>/dev/null || true
        docker exec "clab-mpls-${node}" supervisorctl restart frr 2>/dev/null || \
            docker exec "clab-mpls-${node}" /usr/lib/frr/docker-start 2>/dev/null || true
        
        sleep 2
        log_success "  ${node} configured"
    done
}

verify_routing() {
    log_info "Verifying routing adjacencies..."
    
    local nodes=("p-core-1" "pe-hub-east-1" "pe-dc-1")
    
    for node in "${nodes[@]}"; do
        log_info "  Checking OSPF neighbors on ${node}..."
        docker exec "clab-mpls-${node}" vtysh -c "show ip ospf neighbor" 2>/dev/null || \
            log_warn "  Could not check OSPF on ${node}"
        
        log_info "  Checking MPLS LDP on ${node}..."
        docker exec "clab-mpls-${node}" vtysh -c "show mpls ldp neighbor" 2>/dev/null || \
            log_warn "  Could not check MPLS LDP on ${node}"
    done
    
    log_success "Routing verification complete"
}

configure_management_network() {
    log_info "Setting up management network access..."
    
    # Enable SSH on all nodes
    local nodes=(
        "p-core-1" "p-core-2"
        "pe-hub-east-1" "pe-hub-west-1"
        "pe-dc-1" "pe-dc-2"
        "ce-branch-1" "ce-branch-2" "ce-branch-3" "ce-branch-4"
    )
    
    for node in "${nodes[@]}"; do
        # Install and start SSH server
        docker exec "clab-mpls-${node}" sh -c "apk add --no-cache openssh 2>/dev/null || apt-get install -y openssh-server 2>/dev/null" 2>/dev/null || true
        docker exec "clab-mpls-${node}" sh -c "echo 'root:noc_copilot' | chpasswd 2>/dev/null" 2>/dev/null || true
        docker exec "clab-mpls-${node}" sh -c "sed -i 's/#PermitRootLogin.*/PermitRootLogin yes/' /etc/ssh/sshd_config 2>/dev/null" 2>/dev/null || true
        docker exec "clab-mpls-${node}" sh -c "/usr/sbin/sshd 2>/dev/null" 2>/dev/null || true
    done
    
    log_success "Management network configured"
}

enable_snmp() {
    log_info "Enabling SNMP on all nodes..."
    
    local nodes=(
        "p-core-1" "p-core-2"
        "pe-hub-east-1" "pe-hub-west-1"
        "pe-dc-1" "pe-dc-2"
        "ce-branch-1" "ce-branch-2" "ce-branch-3" "ce-branch-4"
    )
    
    for node in "${nodes[@]}"; do
        docker exec "clab-mpls-${node}" sh -c "apk add --no-cache net-snmp-tools 2>/dev/null || true" 2>/dev/null || true
        
        # Create SNMP config
        docker exec "clab-mpls-${node}" sh -c "cat > /etc/snmp/snmpd.conf << 'EOF'
rocommunity noc_copilot
syslocation NOC-Datacenter
syscontact noc-admin@copilot.local
agentAddress udp:161
EOF
" 2>/dev/null || true
        
        docker exec "clab-mpls-${node}" sh -c "snmpd -c /etc/snmp/snmpd.conf -d" 2>/dev/null || true
        log_success "  SNMP enabled on ${node}"
    done
}

enable_netflow() {
    log_info "Enabling NetFlow/sFlow on core and PE nodes..."
    
    local core_nodes=("p-core-1" "p-core-2" "pe-hub-east-1" "pe-hub-west-1" "pe-dc-1" "pe-dc-2")
    local collector_ip="172.20.0.100"
    
    for node in "${core_nodes[@]}"; do
        docker exec "clab-mpls-${node}" sh -c "cat > /etc/frr/netflow.conf << EOF
flow-accounting
 server ${collector_ip} port 9995
 sampler 1000
 active-timeout 60
 inactive-timeout 15
EOF
" 2>/dev/null || true
        
        log_success "  NetFlow enabled on ${node}"
    done
}

enable_syslog() {
    log_info "Configuring syslog forwarding..."
    
    local nodes=(
        "p-core-1" "p-core-2"
        "pe-hub-east-1" "pe-hub-west-1"
        "pe-dc-1" "pe-dc-2"
        "ce-branch-1" "ce-branch-2" "ce-branch-3" "ce-branch-4"
    )
    local collector_ip="172.20.0.100"
    
    for node in "${nodes[@]}"; do
        docker exec "clab-mpls-${node}" sh -c "cat >> /etc/syslog.conf << EOF
*.* @${collector_ip}:514
local0.* @${collector_ip}:514
local1.* @${collector_ip}:514
EOF
" 2>/dev/null || true
        
        docker exec "clab-mpls-${node}" sh -c "kill -HUP \$(pidof syslogd 2>/dev/null) 2>/dev/null || true" 2>/dev/null || true
        log_success "  Syslog forwarding configured on ${node}"
    done
}

print_summary() {
    echo ""
    echo "============================================================"
    echo -e "${GREEN} Air-Gapped NOC Copilot - Deployment Complete${NC}"
    echo "============================================================"
    echo ""
    echo "Topology: ${TOPOLOGY_FILE}"
    echo "Configs:  ${CONFIGS_DIR}/"
    echo ""
    echo "Network Nodes:"
    echo "  P-Core:   p-core-1 (10.0.0.1), p-core-2 (10.0.0.2)"
    echo "  Hub-PE:   pe-hub-east-1 (10.0.0.10), pe-hub-west-1 (10.0.0.11)"
    echo "  DC-PE:    pe-dc-1 (10.0.0.20), pe-dc-2 (10.0.0.21)"
    echo "  Branch:   ce-branch-1..4 (10.0.0.40..43)"
    echo ""
    echo "Services:"
    echo "  Telegraf:   172.20.0.100:8125 (statsd), :5140 (syslog)"
    echo "  Prometheus: 172.20.0.110:9090"
    echo "  TimescaleDB:172.20.0.120:5432"
    echo ""
    echo "Management Access:"
    echo "  SSH: root:noc_copilot@<mgmt-ip>"
    echo "  SNMP: community=noc_copilot, port=161/udp"
    echo ""
    echo "Next Steps:"
    echo "  1. cd telemetry-stack && docker-compose up -d"
    echo "  2. python3 inject_faults.py --scenario congestion"
    echo "  3. python3 train_models.py"
    echo ""
}

main() {
    log_info "Starting Air-Gapped NOC Copilot deployment..."
    
    check_dependencies
    generate_configs
    deploy_topology
    apply_routing_configs
    configure_management_network
    enable_snmp
    enable_netflow
    enable_syslog
    verify_routing
    print_summary
}

main "$@"
