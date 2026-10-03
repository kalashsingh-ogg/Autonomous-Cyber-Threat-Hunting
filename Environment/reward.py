from .actions import Action


# ============================================================
# HIGH-SEVERITY THREATS
# ============================================================

HIGH_SEVERITY_THREATS = {
    "DDoS-ACK_Fragmentation",
    "DDoS-HTTP_Flood",
    "DDoS-ICMP_Flood",
    "DDoS-ICMP_Fragmentation",
    "DDoS-PSHACK_Flood",
    "DDoS-RSTFINFlood",
    "DDoS-SYN_Flood",
    "DDoS-SlowLoris",
    "DDoS-SynonymousIP_Flood",
    "DDoS-TCP_Flood",
    "DDoS-UDP_Flood",
    "DDoS-UDP_Fragmentation",
}


# ============================================================
# MEDIUM-SEVERITY THREATS
# ============================================================

MEDIUM_SEVERITY_THREATS = {
    "DoS-HTTP_Flood",
    "DoS-SYN_Flood",
    "DoS-TCP_Flood",
    "DoS-UDP_Flood",

    "Mirai-greeth_flood",
    "Mirai-greip_flood",
    "Mirai-udpplain",

    "Recon-HostDiscovery",
    "Recon-OSScan",
    "Recon-PortScan",
}


# ============================================================
# INVESTIGATION-FOCUSED THREATS
# ============================================================

INVESTIGATION_THREATS = {
    "Backdoor_Malware",
    "BrowserHijacking",
    "CommandInjection",
    "DNS_Spoofing",
    "DictionaryBruteForce",
    "MITM-ArpSpoofing",
    "Recon-PingSweep",
    "SqlInjection",
    "Uploading_Attack",
    "VulnerabilityScan",
    "XSS",
}


# ============================================================
# LOW-SEVERITY / BENIGN
# ============================================================

LOW_SEVERITY_THREATS = {
    "BenignTraffic",
}


# ============================================================
# THREAT CATEGORY
# ============================================================

def get_threat_category(true_label):

    if true_label in HIGH_SEVERITY_THREATS:
        return "HIGH"

    if true_label in MEDIUM_SEVERITY_THREATS:
        return "MEDIUM"

    if true_label in INVESTIGATION_THREATS:
        return "INVESTIGATION"

    if true_label in LOW_SEVERITY_THREATS:
        return "LOW"

    return "UNKNOWN"


# ============================================================
# REWARD FUNCTION
# ============================================================

def calculate_reward(
    threat_probability,
    action,
    true_label=None,
    predicted_label=None,
):

    threat_category = get_threat_category(
        true_label
    )

    # ========================================================
    # HIGH-SEVERITY
    # ========================================================

    if threat_category == "HIGH":

        if action == Action.ISOLATE_HOST:
            return 10.0

        if action == Action.BLOCK_IP:
            return 8.0

        if action == Action.INVESTIGATE:
            return 5.0

        if action == Action.INCREASE_MONITORING:
            return 3.0

        if action == Action.MONITOR:
            return -5.0

        if action == Action.ALLOW:
            return -10.0

    # ========================================================
    # MEDIUM-SEVERITY
    # ========================================================

    elif threat_category == "MEDIUM":

        if action == Action.BLOCK_IP:
            return 8.0

        if action == Action.INVESTIGATE:
            return 6.0

        if action == Action.INCREASE_MONITORING:
            return 4.0

        if action == Action.ISOLATE_HOST:
            return 3.0

        if action == Action.MONITOR:
            return -2.0

        if action == Action.ALLOW:
            return -6.0

    # ========================================================
    # INVESTIGATION
    # ========================================================

    elif threat_category == "INVESTIGATION":

        if action == Action.INVESTIGATE:
            return 8.0

        if action == Action.INCREASE_MONITORING:
            return 5.0

        if action == Action.BLOCK_IP:
            return 3.0

        if action == Action.ISOLATE_HOST:
            return 2.0

        if action == Action.MONITOR:
            return -1.0

        if action == Action.ALLOW:
            return -5.0

    # ========================================================
    # BENIGN
    # ========================================================

    elif threat_category == "LOW":

        if action == Action.ALLOW:
            return 5.0

        if action == Action.MONITOR:
            return 4.0

        if action == Action.INVESTIGATE:
            return -2.0

        if action == Action.INCREASE_MONITORING:
            return -2.0

        if action == Action.BLOCK_IP:
            return -6.0

        if action == Action.ISOLATE_HOST:
            return -8.0

    # ========================================================
    # UNKNOWN
    # ========================================================

    else:

        if action == Action.MONITOR:
            return 1.0

        return 0.0

    return 0.0