from enum import IntEnum


class Action(IntEnum):
    MONITOR = 0
    BLOCK_IP = 1
    ISOLATE_HOST = 2
    INCREASE_MONITORING = 3
    INVESTIGATE = 4
    ALLOW = 5


ACTION_NAMES = {
    Action.MONITOR: "Monitor",
    Action.BLOCK_IP: "Block IP",
    Action.ISOLATE_HOST: "Isolate Host",
    Action.INCREASE_MONITORING: "Increase Monitoring",
    Action.INVESTIGATE: "Investigate",
    Action.ALLOW: "Allow",
}