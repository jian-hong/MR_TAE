import { useEffect, useState } from "react";
import { getJson } from "../api.js";

function renderTree(nodes, depth = 0) {
  if (!nodes) return null;
  return (
    <ul style={{ listStyle: "none", paddingLeft: depth ? 16 : 0 }}>
      {nodes.map((n) => (
        <li key={n.path} className="mono" style={{ fontSize: 12 }}>
          {n.type === "dir" ? "[d] " : "[f] "}
          {n.name}
          {n.children ? renderTree(n.children, depth + 1) : null}
        </li>
      ))}
    </ul>
  );
}

export default function Files() {
  const [tree, setTree] = useState(null);
  useEffect(() => {
    getJson("/api/files").then(setTree).catch(() => setTree([]));
  }, []);
  return (
    <div>
      <h1>Files (results/)</h1>
      <div className="card">{renderTree(tree)}</div>
    </div>
  );
}
