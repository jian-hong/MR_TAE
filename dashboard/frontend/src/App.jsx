import { NavLink, Route, Routes } from "react-router-dom";
import Overview from "./pages/Overview.jsx";
import Training from "./pages/Training.jsx";
import GPU from "./pages/GPU.jsx";
import Benchmark from "./pages/Benchmark.jsx";
import Files from "./pages/Files.jsx";
import Logs from "./pages/Logs.jsx";

export default function App() {
  return (
    <div className="layout">
      <nav>
        <h3 style={{ marginTop: 0 }}>AE-PD</h3>
        <NavLink to="/">Overview</NavLink>
        <NavLink to="/training">Training</NavLink>
        <NavLink to="/gpu">GPU</NavLink>
        <NavLink to="/benchmark">Benchmark</NavLink>
        <NavLink to="/files">Files</NavLink>
        <NavLink to="/logs">Logs</NavLink>
      </nav>
      <main>
        <Routes>
          <Route path="/" element={<Overview />} />
          <Route path="/training" element={<Training />} />
          <Route path="/gpu" element={<GPU />} />
          <Route path="/benchmark" element={<Benchmark />} />
          <Route path="/files" element={<Files />} />
          <Route path="/logs" element={<Logs />} />
        </Routes>
      </main>
    </div>
  );
}
