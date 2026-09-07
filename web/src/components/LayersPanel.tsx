import { useMemo, useState } from "react";
import type { Asset } from "../types";
import { CLASS_COLORS, CLASS_LABELS } from "../types";

interface Props {
  assets: Asset[];
  visibleClasses: Set<string>;
  onToggleClass: (cls: string) => void;
  selectedId: string | null;
  onSelect: (id: string) => void;
}

type Tab = "layers" | "twin";

const CATEGORIES: { label: string; classes: string[] }[] = [
  { label: "Pavement", classes: ["pavement", "pavement_marking"] },
  {
    label: "Utilities",
    classes: ["utility_pole", "overhead_conductor", "utility_cabinet"],
  },
  { label: "Signs", classes: ["traffic_sign"] },
  {
    label: "Safety",
    classes: ["guardrail", "safety_barrier", "rumble_strip"],
  },
];

export default function LayersPanel({
  assets,
  visibleClasses,
  onToggleClass,
  selectedId,
  onSelect,
}: Props) {
  const [tab, setTab] = useState<Tab>("layers");

  const counts = useMemo(() => {
    const map = new Map<string, number>();
    for (const asset of assets) {
      map.set(asset.class, (map.get(asset.class) ?? 0) + 1);
    }
    return map;
  }, [assets]);

  const present = useMemo(() => new Set(assets.map((a) => a.class)), [assets]);

  const byClass = useMemo(() => {
    const map = new Map<string, Asset[]>();
    for (const asset of assets) {
      const list = map.get(asset.class) ?? [];
      list.push(asset);
      map.set(asset.class, list);
    }
    return map;
  }, [assets]);

  const allOn =
    [...present].length > 0 && [...present].every((cls) => visibleClasses.has(cls));

  return (
    <aside className="sidebar">
      <div className="sb-head">
        <h3>{tab === "layers" ? "Asset layers" : "Digital twin"}</h3>
        <div className="mode-switch" style={{ border: "none" }}>
          <button className={tab === "layers" ? "on" : ""} onClick={() => setTab("layers")}>
            Layers
          </button>
          <button className={tab === "twin" ? "on" : ""} onClick={() => setTab("twin")}>
            Twin
          </button>
        </div>
      </div>

      <div className="sb-body">
        {tab === "layers" ? (
          <>
            <div className="layer-row" onClick={() => {
              for (const cls of present) {
                if (allOn) {
                  if (visibleClasses.has(cls)) onToggleClass(cls);
                } else if (!visibleClasses.has(cls)) {
                  onToggleClass(cls);
                }
              }
            }}>
              <span className="check">{allOn ? "☑" : "☐"}</span>
              <span className="lname" style={{ letterSpacing: 1.2, textTransform: "uppercase", fontSize: 11 }}>
                {allOn ? "Hide all" : "Show all"}
              </span>
              <span className="count">{assets.length}</span>
            </div>

            {CATEGORIES.filter((cat) => cat.classes.some((c) => present.has(c))).map((cat) => {
              const presentClasses = cat.classes.filter((c) => present.has(c));
              const catOn = presentClasses.every((c) => visibleClasses.has(c));
              const catCount = presentClasses.reduce((sum, c) => sum + (counts.get(c) ?? 0), 0);
              return (
                <div key={cat.label}>
                  <div
                    className="layer-row cat-row"
                    onClick={() => {
                      for (const cls of presentClasses) {
                        if (catOn ? visibleClasses.has(cls) : !visibleClasses.has(cls)) {
                          onToggleClass(cls);
                        }
                      }
                    }}
                  >
                    <span className="check">{catOn ? "☑" : "☐"}</span>
                    <span className="lname">{cat.label}</span>
                    <span className="count">{catCount}</span>
                  </div>
                  {presentClasses.map((cls) => (
                    <div key={cls} className={`layer-row ${visibleClasses.has(cls) ? "" : "off"}`} onClick={() => onToggleClass(cls)}>
                      <span className="check">{visibleClasses.has(cls) ? "☑" : "☐"}</span>
                      <span className="swatch" style={{ background: CLASS_COLORS[cls] ?? "#8899aa" }} />
                      <span className="lname">{CLASS_LABELS[cls] ?? cls}</span>
                      <span className="count">{counts.get(cls) ?? 0}</span>
                    </div>
                  ))}
                </div>
              );
            })}
          </>
        ) : (
          <>
            <div className="twin-root">ROAD</div>
            {[...byClass.entries()]
              .sort(([a], [b]) => a.localeCompare(b))
              .map(([cls, list]) => (
                <div key={cls}>
                  <div className="twin-group">
                    {CLASS_LABELS[cls] ?? cls} · {list.length}
                  </div>
                  {list.map((asset) => (
                    <div
                      key={asset.asset_id}
                      className={`twin-item ${selectedId === asset.asset_id ? "selected" : ""}`}
                      onClick={() => onSelect(asset.asset_id)}
                    >
                      <span
                        className="dot"
                        style={{ background: CLASS_COLORS[cls] ?? "#8899aa" }}
                      />
                      {asset.asset_id}
                      <span style={{ marginLeft: "auto", color: "var(--text-faint)", fontSize: 10.5 }}>
                        {Math.round(asset.confidence * 100)}%
                      </span>
                    </div>
                  ))}
                </div>
              ))}
          </>
        )}
      </div>
    </aside>
  );
}