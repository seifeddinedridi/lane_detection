from dataclasses import dataclass
from enum import Enum, auto


class CityScapeLabel(Enum):
    # Flat
    ROAD = auto()
    SIDEWALK = auto()
    PARKING = auto()
    RAIL_TRACK = auto()
    # Human
    PERSON = auto()
    RIDER = auto()
    # Vehicle
    CAR = auto()
    TRUCK = auto()
    BUS = auto()
    ON_RAILS = auto()
    MOTORCYCLE = auto()
    BICYCLE = auto()
    CARAVAN = auto()
    TRAILER = auto()
    # Construction
    BUILDING = auto()
    WALL = auto()
    FENCE = auto()
    GUARD_RAIL = auto()
    BRIDGE = auto()
    TUNNEL = auto()
    # Object
    POLE = auto()
    POLE_GROUP = auto()
    TRAFFIC_SIGN = auto()
    TRAFFIC_LIGHT = auto()
    # Nature
    VEGETATION = auto()
    TERRAIN = auto()
    # Sky
    SKY = auto()
    # Void
    GROUND = auto()
    DYNAMIC = auto()
    STATIC = auto()
    EGO_VEHICLE = auto()
    OUT_OF_ROI = auto()
    RECTIFICATION_BORDER = auto()
    UNLABELED = auto()

    @staticmethod
    def parse(label):
        label = label.replace(' ', '_').upper()
        for v in CityScapeLabel:
            if v.name == label:
                return v
        raise Exception(f'Unsupported label {label}')


@dataclass
class CityScapeRegion:
    label: CityScapeLabel
    polygon: list[tuple]


@dataclass
class CityScapeAnnotatedImage:
    imgWidth: int
    imgHeight: int
    objects: list[CityScapeRegion]
