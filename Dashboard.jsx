import React from "react";

export default function Dashboard() {
  return (
    <div className="flex h-screen bg-gray-100 font-sans">
      
      {/* Sidebar */}
      <div className="w-60 bg-slate-900 text-white p-5">
        <h1 className="text-xl font-bold mb-8">BlueWave AI</h1>
        <nav className="space-y-4">
          <p className="hover:text-blue-400 cursor-pointer">🧪 Input Config</p>
          <p className="hover:text-blue-400 cursor-pointer">🤖 Models</p>
          <p className="hover:text-blue-400 cursor-pointer">📊 Results</p>
          <p className="hover:text-blue-400 cursor-pointer">🧠 Insights</p>
          <p className="hover:text-blue-400 cursor-pointer">⚙️ Settings</p>
        </nav>
      </div>

      {/* Main Content */}
      <div className="flex-1 flex flex-col">
        
        {/* Navbar */}
        <div className="h-16 bg-slate-800 text-white flex items-center justify-between px-6">
          <h2 className="text-lg font-semibold">PET Intelligence Dashboard</h2>
          <div>👤</div>
        </div>

        {/* Content */}
        <div className="p-6 space-y-6 overflow-y-auto">

          {/* Input Panel */}
          <div className="bg-white p-5 rounded-xl shadow">
            <h3 className="font-semibold mb-3">Input Data</h3>
            <div className="grid grid-cols-3 gap-4">
              <input className="border p-2 rounded" placeholder="PET Formulation" />
              <input className="border p-2 rounded" placeholder="Processing Conditions" />
              <input className="border p-2 rounded" placeholder="Ask a question..." />
            </div>
            <button className="mt-4 bg-blue-600 text-white px-4 py-2 rounded">
              Run Analysis
            </button>
          </div>

          {/* Model Layer */}
          <div className="grid grid-cols-4 gap-4">
            <ModelCard title="Env Model 🌱" color="bg-green-100" desc="CO₂, Energy" />
            <ModelCard title="Degradation 🔁" color="bg-orange-100" desc="IV, Color, Strength" />
            <ModelCard title="Property ⚙️" color="bg-purple-100" desc="Tg, MFI, Tensile" />
            <ModelCard title="NL Query 💬" color="bg-yellow-100" desc="Ask + Answer" />
          </div>

          {/* Output Dashboard */}
          <div className="grid grid-cols-4 gap-4">
            <OutputCard title="CO₂ + Energy" value="2.3 kg / 45 MJ" />
            <OutputCard title="IV per Cycle" value="0.76 → 0.62" />
            <OutputCard title="Properties" value="Tg: 75°C" />
            <OutputCard title="Answer" value="High barrier grade recommended" />
          </div>

          {/* Application Layer */}
          <div className="grid grid-cols-4 gap-4">
            <AppCard title="Optimizer" desc="Minimize rPET" />
            <AppCard title="Recyclability" desc="Cycles before failure" />
            <AppCard title="Grade Selector" desc="Best material grade" />
            <AppCard title="R&D Assistant" desc="Ask anything" />
          </div>

        </div>
      </div>
    </div>
  );
}

/* Reusable Components */

function ModelCard({ title, desc, color }) {
  return (
    <div className={`p-4 rounded-xl shadow ${color}`}>
      <h4 className="font-semibold">{title}</h4>
      <p className="text-sm mt-2">{desc}</p>
    </div>
  );
}

function OutputCard({ title, value }) {
  return (
    <div className="bg-white p-4 rounded-xl shadow">
      <h4 className="text-sm text-gray-500">{title}</h4>
      <p className="text-lg font-semibold mt-2">{value}</p>
    </div>
  );
}

function AppCard({ title, desc }) {
  return (
    <div className="bg-white p-4 rounded-xl shadow hover:shadow-lg cursor-pointer">
      <h4 className="font-semibold">{title}</h4>
      <p className="text-sm text-gray-500 mt-2">{desc}</p>
    </div>
  );
}