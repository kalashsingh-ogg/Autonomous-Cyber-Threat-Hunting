from dataclasses import dataclass


@dataclass
class CyberState:
    threat_class: int = 0
    threat_probability: float = 0.0
    host_risk: float = 0.0
    connection_risk: float = 0.0
    monitoring_level: float = 0.5
    host_isolated: float = 0.0
    ip_blocked: float = 0.0
    investigation_active: float = 0.0

    def to_vector(self):
        return [
            float(self.threat_class),
            float(self.threat_probability),
            float(self.host_risk),
            float(self.connection_risk),
            float(self.monitoring_level),
            float(self.host_isolated),
            float(self.ip_blocked),
            float(self.investigation_active),
        ]