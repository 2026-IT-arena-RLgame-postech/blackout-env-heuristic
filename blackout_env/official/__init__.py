"""The official BlackOut observation format (ist-tech-AI-games/blackout @ d2220a7) and its reverse.

The submission runtime only feeds the *official* raw observations:

* per unit agent, ``vector`` float32[45] (``BlackOutAgent.CollectObservations``)::

      [4u..4u+3]  unit u = 0..9: pos_x, pos_y (= pos / 24, in [0, 1]), team sign (+1 ally /
                  -1 enemy of the observer), holding item id (0 none, 1 battery, 2-5 specials)
      [40]        the observer's own class id (Collector 0 / Hunter 1 / Carrier 2)
      [41], [42]  own / opponent score / 100
      [43]        episode time left (1 - EpisodeTimer.Ratio)
      [44]        the observer's unitIndex

* ``graphic`` uint8 ids [96, 96] (``SemanticMapRenderer``, 4 px per tile, team A view, rows
  top-down): 0 empty, 1 wall, 2 ally storage, 3 enemy storage, 4 ally unit, 5 enemy unit,
  6 + item index (6 battery, 7-10 specials) -- one pixel per ground item (tile centre) and per
  unit (written last, so a unit can hide an item pixel).

Our policies consume the richer fork format (MapObsAgent float32[44] + packed 24x24 map).  The
two differ only by what the official one leaves out: battery *amounts* (held and on tiles), the
classes of the other nine units, the absorption timer and the site / spawn tile categories.
``OfficialTracker`` recovers all of them by driving the rule-exact simulator with the observed
positions; the field battery amounts are exact once ``seed_search`` finds the reset seed.
"""

from .render import (from_preprocessed, official_map, official_preprocessed, official_vector, parse_vector,
                     team_of_preprocessed)
from .tracker import OfficialTracker

__all__ = ["OfficialTracker", "from_preprocessed", "official_map", "official_preprocessed", "official_vector",
           "parse_vector", "team_of_preprocessed"]
