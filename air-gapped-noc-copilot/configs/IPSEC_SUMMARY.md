# SD-WAN IPSec Overlay Configuration Summary

## Topology

```
                  [Hub-East]                    [Hub-West]
                  (pe-hub-east-1)               (pe-hub-west-1)
                     /    \                       /    \
                    /      \                     /      \
            IPSec / IPSec  IPSec / IPSec  IPSec / IPSec  IPSec / IPSec
                 /          \                     /          \
                /            \                   /            \
        [Branch-1]      [Branch-2]        [Branch-3]      [Branch-4]
        (ce-branch-1)   (ce-branch-2)     (ce-branch-3)   (ce-branch-4)
```

## Tunnel Details

| Tunnel | Local | Remote | SPI | Encryption | Lifetime |
|--------|-------|--------|-----|------------|----------|
| hub-east<->branch-1 | 10.255.0.10 | 10.255.0.40 | 0xA001 | aes-256-gcm | 3600s |
| hub-east<->branch-2 | 10.255.0.10 | 10.255.0.41 | 0xA002 | aes-256-gcm | 3600s |
| hub-west<->branch-3 | 10.255.0.11 | 10.255.0.42 | 0xA003 | aes-256-gcm | 3600s |
| hub-west<->branch-4 | 10.255.0.11 | 10.255.0.43 | 0xA004 | aes-256-gcm | 3600s |
| branch-1<->hub-east | 10.255.0.40 | 10.255.0.10 | 0xB001 | aes-256-gcm | 3600s |
| branch-2<->hub-east | 10.255.0.41 | 10.255.0.10 | 0xB002 | aes-256-gcm | 3600s |
| branch-3<->hub-west | 10.255.0.42 | 10.255.0.11 | 0xB003 | aes-256-gcm | 3600s |
| branch-4<->hub-west | 10.255.0.43 | 10.255.0.11 | 0xB004 | aes-256-gcm | 3600s |

## IP Addressing

| Device | Loopback | Tunnel Local | Subnet |
|--------|----------|--------------|--------|
| pe-hub-east-1 | 10.255.0.10 | 172.16.1.1 | 10.0.1.0/24 |
| pe-hub-east-1 | 10.255.0.10 | 172.16.2.1 | 10.0.1.0/24 |
| pe-hub-west-1 | 10.255.0.11 | 172.16.3.1 | 10.0.2.0/24 |
| pe-hub-west-1 | 10.255.0.11 | 172.16.4.1 | 10.0.2.0/24 |
| ce-branch-1 | 10.255.0.40 | 172.16.1.2 | 10.0.11.0/24 |
| ce-branch-2 | 10.255.0.41 | 172.16.2.2 | 10.0.12.0/24 |
| ce-branch-3 | 10.255.0.42 | 172.16.3.2 | 10.0.13.0/24 |
| ce-branch-4 | 10.255.0.43 | 172.16.4.2 | 10.0.14.0/24 |