import React from "react";
import ReactDOM from "react-dom/client";
import App from "./App";
import "./index.css";   // 🔴 这一行必须存在

ReactDOM.createRoot(document.getElementById("root")!).render(
  <React.StrictMode>
    <App />
  </React.StrictMode>
);
