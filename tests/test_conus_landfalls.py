"""
Unit tests for calibration/filter_conus_landfalls.py.

Core focus: _first_intersection with MultiLineString boundary.
The FL script only needed LinearRing boundaries (fl_poly.boundary).
land_union.boundary on the CONUS MultiPolygon IS a MultiLineString —
mainland + Keys + barrier islands + Long Island each contribute a ring.
If the MultiLineString branch is wrong, detection silently returns empty
crossings with no error.
"""

import pytest
from shapely.geometry import LineString, MultiLineString, Point

from calibration.filter_conus_landfalls import _first_intersection


class TestFirstIntersectionMultiLineString:
    """_first_intersection must handle MultiLineString boundaries explicitly."""

    def test_single_component_crossing(self):
        """Segment crosses one component of a two-part MultiLineString."""
        # coast1: y=1, x in [0,2]; coast2: y=1, x in [3,5] (gap between them)
        boundary = MultiLineString([[(0, 1), (2, 1)], [(3, 1), (5, 1)]])
        seg = LineString([(1, 0), (1, 2)])  # vertical — crosses coast1 at (1,1)
        pt = _first_intersection(seg, boundary)
        assert pt is not None
        assert abs(pt.x - 1.0) < 1e-9
        assert abs(pt.y - 1.0) < 1e-9

    def test_picks_closest_to_segment_start(self):
        """When two components are crossed, returns the crossing closest to start."""
        # Two vertical lines at x=1 and x=3
        boundary = MultiLineString([[(1, 0), (1, 4)], [(3, 0), (3, 4)]])
        seg = LineString([(0, 2), (4, 2)])  # horizontal — crosses at x=1 then x=3
        pt = _first_intersection(seg, boundary)
        assert pt is not None
        assert abs(pt.x - 1.0) < 1e-9, f"Expected x≈1.0 (first crossing), got x={pt.x}"

    def test_returns_second_when_first_not_crossed(self):
        """Returns second component crossing if only that component is crossed."""
        boundary = MultiLineString([[(0, 5), (2, 5)], [(3, 1), (5, 1)]])
        seg = LineString([(4, 0), (4, 2)])  # only crosses coast2 at (4,1)
        pt = _first_intersection(seg, boundary)
        assert pt is not None
        assert abs(pt.x - 4.0) < 1e-9
        assert abs(pt.y - 1.0) < 1e-9

    def test_no_crossing_returns_none(self):
        """Segment parallel to both components returns None."""
        boundary = MultiLineString([[(0, 2), (4, 2)], [(0, 3), (4, 3)]])
        seg = LineString([(0, 0), (4, 0)])  # no intersection
        pt = _first_intersection(seg, boundary)
        assert pt is None

    def test_three_component_picks_first(self):
        """With three crossings, the one closest to segment start is returned."""
        # Vertical lines at x=1, x=3, x=5
        boundary = MultiLineString([
            [(1, 0), (1, 4)],
            [(3, 0), (3, 4)],
            [(5, 0), (5, 4)],
        ])
        seg = LineString([(0, 2), (6, 2)])  # crosses all three
        pt = _first_intersection(seg, boundary)
        assert pt is not None
        assert abs(pt.x - 1.0) < 1e-9, f"Expected first crossing at x=1, got x={pt.x}"


class TestFirstIntersectionExistingTypes:
    """Existing behavior (LinearRing / LineString / Point results) is preserved."""

    def test_simple_linestring_boundary(self):
        """Segment crossing a plain LineString boundary returns the crossing point."""
        boundary = LineString([(0, 1), (4, 1)])
        seg = LineString([(2, 0), (2, 2)])
        pt = _first_intersection(seg, boundary)
        assert pt is not None
        assert abs(pt.x - 2.0) < 1e-9
        assert abs(pt.y - 1.0) < 1e-9

    def test_empty_intersection_returns_none(self):
        """Parallel segment returns None."""
        boundary = LineString([(0, 1), (4, 1)])
        seg = LineString([(0, 2), (4, 2)])
        pt = _first_intersection(seg, boundary)
        assert pt is None

    def test_multipoint_result_picks_first(self):
        """When intersection yields multiple points, returns closest to start."""
        # Two separate horizontal boundaries at y=1 and y=3, both crossed
        from shapely.geometry import MultiPoint
        # Construct scenario: a zigzag line that crosses a horizontal boundary twice.
        # Easier: use two parallel horizontal lines (MultiLineString) that a vertical
        # segment crosses.
        boundary = MultiLineString([[(0, 1), (4, 1)], [(0, 3), (4, 3)]])
        seg = LineString([(2, 0), (2, 4)])  # crosses at y=1 then y=3
        pt = _first_intersection(seg, boundary)
        assert pt is not None
        assert abs(pt.y - 1.0) < 1e-9, f"Expected y≈1.0 (lower crossing first), got y={pt.y}"
