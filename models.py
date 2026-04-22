from dataclasses import dataclass, asdict
from typing import Optional, List, Dict, Any


@dataclass
class ChannelResult:
    label: str
    value: Optional[float]
    unit: Optional[str]
    raw_reading: Optional[str]
    confidence: float

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class DeviceResult:
    image_path: str
    device_type: str
    display_type: str
    detected_text: Optional[str]
    value: Optional[float]
    unit: Optional[str]
    raw_reading: Optional[str]
    decimal_point: Optional[bool]
    confidence: float
    status: str
    channels: Optional[List[Dict[str, Any]]] = None
    extra: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
