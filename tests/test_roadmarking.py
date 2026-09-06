from __future__ import annotations

from pathlib import Path

from infra_inventory.roadmarking import parse_dxf_outputs

DXF_LINE = """0
SECTION
2
ENTITIES
0
LINE
8
marking
10
0.0
20
0.0
30
0.0
11
5.0
21
0.0
31
0.0
0
LINE
8
marking
10
5.0
20
0.0
30
0.0
11
10.0
21
0.0
31
0.0
0
LINE
8
other
10
50.0
20
0.0
30
0.0
11
51.0
21
0.0
31
0.0
0
ENDSEC
0
EOF
"""

DXF_POLYLINE = """0
SECTION
2
ENTITIES
0
LWPOLYLINE
8
lane
90
3
10
0.0
20
5.0
30
0.0
10
1.0
20
5.0
30
0.0
10
2.0
20
5.0
30
0.0
0
ENDSEC
0
EOF
"""


def test_parse_line_entities(tmp_path: Path) -> None:
    output = tmp_path / "output"
    output.mkdir()
    (output / "markings.dxf").write_text(DXF_LINE)
    instances = parse_dxf_outputs(output)
    assert len(instances) == 2  # two connected segments group; the far one is separate
    groups = sorted(instances, key=lambda item: item["length_m"], reverse=True)
    assert groups[0]["length_m"] >= 10.0
    assert groups[1]["length_m"] < 2.0
    assert all(instance["detection_method"] == "roadmarkingextraction-v1" for instance in instances)


def test_parse_polyline_entities(tmp_path: Path) -> None:
    output = tmp_path / "output"
    output.mkdir()
    (output / "lane.dxf").write_text(DXF_POLYLINE)
    instances = parse_dxf_outputs(output)
    assert len(instances) == 1
    assert instances[0]["length_m"] >= 2.0


def test_empty_output_dir(tmp_path: Path) -> None:
    assert parse_dxf_outputs(tmp_path / "missing") == []
    (tmp_path / "empty").mkdir()
    assert parse_dxf_outputs(tmp_path / "empty") == []