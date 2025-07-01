"use client";

import { useState } from "react";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Slider } from "@/components/ui/slider";
import { Alert, AlertDescription } from "@/components/ui/alert";
import { Download, Play, Loader2 } from "lucide-react";

interface Exit {
  x: number;
  y: number;
}

interface SimulationConfig {
  grid_rows: number;
  grid_cols: number;
  num_agents: number;
  fire_growth_rate: number;
  follower_percentage: number;
  no_repulsion_percentage: number;
  crowded_exit_avoider_percentage: number;
  num_obstacles: number;
  agent_speed: number;
  exits?: Exit[];
}

interface SimulationStats {
  total_agents: number;
  escaped: number;
  dead: number;
  followers: number;
  no_repulsion: number;
  avoiders: number;
}

export default function SimulationPage() {
  const [config, setConfig] = useState<SimulationConfig>({
    grid_rows: 25,
    grid_cols: 25,
    num_agents: 50,
    fire_growth_rate: 0.03,
    follower_percentage: 20,
    no_repulsion_percentage: 20,
    crowded_exit_avoider_percentage: 30,
    num_obstacles: 15,
    agent_speed: 0.015,
    exits: [],
  });

  const [simulationId, setSimulationId] = useState<string | null>(null);
  const [status, setStatus] = useState<
    "idle" | "running" | "completed" | "error"
  >("idle");
  const [stats, setStats] = useState<SimulationStats | null>(null);
  const [error, setError] = useState<string | null>(null);

  const API_BASE = process.env.NEXT_PUBLIC_API_BASE!;

  const startSimulation = async () => {
    try {
      setStatus("running");
      setError(null);
      setStats(null);

      const response = await fetch(`${API_BASE}/api/simulate`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(config),
      });

      if (!response.ok) throw new Error("Failed to start simulation");

      const data = await response.json();
      setSimulationId(data.simulation_id);
      pollStatus(data.simulation_id);
    } catch (err) {
      setStatus("error");
      setError(err instanceof Error ? err.message : "Unknown error");
    }
  };

  const pollStatus = async (id: string) => {
    try {
      const response = await fetch(`${API_BASE}/api/status/${id}`);
      const data = await response.json();

      if (data.status === "completed") {
        setStatus("completed");
        setStats(data.stats);
      } else if (data.status === "error") {
        setStatus("error");
        setError(data.error || "Simulation failed");
      } else {
        setTimeout(() => pollStatus(id), 2000);
      }
    } catch {
      setStatus("error");
      setError("Failed to check simulation status");
    }
  };

  const downloadSimulation = () => {
    if (simulationId) {
      window.open(`${API_BASE}/api/download/${simulationId}`, "_blank");
    }
  };

  const updateConfig = (key: keyof SimulationConfig, value: number) => {
    setConfig((prev) => ({ ...prev, [key]: value }));
  };

  return (
    <div className="min-h-screen bg-gray-100 p-4">
      <div className="max-w-6xl mx-auto">
        {/* Header */}
        <div className="text-center space-y-1 mb-8">
          <img src="/tuelogo.svg" alt="TU/e Logo" className="mx-auto h-16" />
          <h1 className="text-2xl font-semibold text-gray-800">
            Innovationspace Bachelor End Project - 0ISBEP05
          </h1>
          <h2 className="text-3xl font-bold text-black">
            Simulation and Control of Swarm Robots
            <br />
            in Panic-based Evacuation Scenarios
          </h2>
          <p className="text-sm text-gray-700">Quartile 3 & 4 – 2025–2026</p>
          <p className="text-sm text-gray-700">
            <strong>Student:</strong> G.M. Volanschi – 1778382 – Computer
            Science and Engineering
          </p>
          <p className="text-sm text-gray-700">
            <strong>Supervisor:</strong> Dr. K. Cuijpers
          </p>
        </div>

        <div className="grid md:grid-cols-2 gap-6">
          {/* Config Panel */}
          <Card className="h-fit shadow-md rounded-2xl">
            <CardHeader>
              <CardTitle className="flex items-center gap-2">
                <div className="w-3 h-3 bg-blue-500 rounded-full"></div>
                Simulation Parameters
              </CardTitle>
            </CardHeader>
            <CardContent className="space-y-6">
              {/* Grid Size */}
              <div className="space-y-4">
                <Label className="text-sm font-semibold text-gray-700">
                  Grid Dimensions
                </Label>
                <div className="grid grid-cols-2 gap-4">
                  <div>
                    <Label
                      htmlFor="grid_rows"
                      className="text-xs text-gray-600"
                    >
                      Rows
                    </Label>
                    <Input
                      id="grid_rows"
                      type="number"
                      value={config.grid_rows}
                      onChange={(e) =>
                        updateConfig(
                          "grid_rows",
                          Math.max(10, parseInt(e.target.value) || 10)
                        )
                      }
                      min={10}
                      max={50}
                      className="mt-1"
                    />
                  </div>
                  <div>
                    <Label
                      htmlFor="grid_cols"
                      className="text-xs text-gray-600"
                    >
                      Columns
                    </Label>
                    <Input
                      id="grid_cols"
                      type="number"
                      value={config.grid_cols}
                      onChange={(e) =>
                        updateConfig(
                          "grid_cols",
                          Math.max(10, parseInt(e.target.value) || 10)
                        )
                      }
                      min={10}
                      max={50}
                      className="mt-1"
                    />
                  </div>
                </div>
              </div>

              {/* Agent Speed */}
              <div className="space-y-2">
                <Label className="text-sm font-semibold text-gray-700">
                  Agent Speed: {config.agent_speed.toFixed(3)}
                </Label>
                <Slider
                  value={[config.agent_speed]}
                  onValueChange={([value]) =>
                    updateConfig("agent_speed", value)
                  }
                  min={0.005}
                  max={0.05}
                  step={0.001}
                  className="w-full"
                />
              </div>

              {/* Number of Agents */}
              <div className="space-y-2">
                <Label className="text-sm font-semibold text-gray-700">
                  Number of Agents: {config.num_agents}
                </Label>
                <Slider
                  value={[config.num_agents]}
                  onValueChange={([value]) => updateConfig("num_agents", value)}
                  min={10}
                  max={50}
                  step={5}
                  className="w-full"
                />
              </div>

              {/* Fire Growth Rate */}
              <div className="space-y-2">
                <Label className="text-sm font-semibold text-gray-700">
                  Fire Growth Rate: {config.fire_growth_rate.toFixed(3)}
                </Label>
                <Slider
                  value={[config.fire_growth_rate]}
                  onValueChange={([value]) =>
                    updateConfig("fire_growth_rate", value)
                  }
                  min={0.01}
                  max={0.1}
                  step={0.005}
                  className="w-full"
                />
              </div>

              {/* Obstacles */}
              <div className="space-y-2">
                <Label className="text-sm font-semibold text-gray-700">
                  Number of Obstacles: {config.num_obstacles}
                </Label>
                <Slider
                  value={[config.num_obstacles]}
                  onValueChange={([value]) =>
                    updateConfig("num_obstacles", value)
                  }
                  min={0}
                  max={50}
                  step={1}
                  className="w-full"
                />
              </div>

              {/* Personalities */}
              <div className="space-y-4">
                <Label className="text-sm font-semibold text-gray-700">
                  Agent Personalities (%) – Can overlap
                </Label>

                {[
                  "follower_percentage",
                  "no_repulsion_percentage",
                  "crowded_exit_avoider_percentage",
                ].map((key) => (
                  <div className="space-y-2" key={key}>
                    <Label className="text-xs text-gray-600 capitalize">
                      {key.replace(/_/g, " ").replace("percentage", "")}:{" "}
                      {config[key as keyof SimulationConfig]}%
                    </Label>
                    <Slider
                      value={[config[key as keyof SimulationConfig] as number]}
                      onValueChange={([value]) =>
                        updateConfig(key as keyof SimulationConfig, value)
                      }
                      min={0}
                      max={100}
                      step={5}
                      className="w-full"
                    />
                  </div>
                ))}
              </div>
              {/* Exit Configuration */}
              <div className="space-y-2">
                <Label className="text-sm font-semibold text-gray-700">
                  Exits (1 to 4)
                </Label>
                {config.exits?.map((exit, index) => (
                  <div key={index} className="grid grid-cols-2 gap-2">
                    <div>
                      <Label className="text-xs text-gray-600">
                        X Position
                      </Label>
                      <Input
                        type="number"
                        min={0}
                        max={config.grid_cols}
                        step={0.1}
                        value={exit.x}
                        onChange={(e) => {
                          const exits = [...(config.exits || [])];
                          exits[index].x = parseFloat(e.target.value) || 0;
                          setConfig((prev) => ({ ...prev, exits }));
                        }}
                      />
                    </div>
                    <div>
                      <Label className="text-xs text-gray-600">
                        Y Position
                      </Label>
                      <Input
                        type="number"
                        min={0}
                        max={config.grid_rows}
                        step={0.1}
                        value={exit.y}
                        onChange={(e) => {
                          const exits = [...(config.exits || [])];
                          exits[index].y = parseFloat(e.target.value) || 0;
                          setConfig((prev) => ({ ...prev, exits }));
                        }}
                      />
                    </div>
                  </div>
                ))}

                <div className="flex gap-2">
                  <Button
                    variant="outline"
                    disabled={config.exits && config.exits.length >= 4}
                    onClick={() =>
                      setConfig((prev) => ({
                        ...prev,
                        exits: [...(prev.exits || []), { x: 0.5, y: 0.5 }],
                      }))
                    }
                  >
                    + Add Exit
                  </Button>
                  <Button
                    variant="ghost"
                    disabled={!config.exits || config.exits.length === 0}
                    onClick={() =>
                      setConfig((prev) => ({
                        ...prev,
                        exits: prev.exits?.slice(0, -1),
                      }))
                    }
                  >
                    Remove Last Exit
                  </Button>
                </div>
              </div>

              {/* Run Button */}
              <Button
                onClick={startSimulation}
                disabled={status === "running"}
                className="w-full bg-gradient-to-r from-red-600 to-red-700 hover:from-red-700 hover:to-red-800"
              >
                {status === "running" ? (
                  <>
                    <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                    Running Simulation...
                  </>
                ) : (
                  <>
                    <Play className="mr-2 h-4 w-4" />
                    Run Simulation
                  </>
                )}
              </Button>
            </CardContent>
          </Card>

          {/* Result Panel */}
          <Card className="h-fit shadow-md rounded-2xl">
            <CardHeader>
              <CardTitle className="flex items-center gap-2">
                <div className="w-3 h-3 bg-green-500 rounded-full"></div>
                Simulation Results
              </CardTitle>
            </CardHeader>
            <CardContent className="space-y-4">
              {status === "idle" && (
                <p className="text-sm text-gray-600">
                  No simulation started yet.
                </p>
              )}
              {status === "running" && (
                <Alert>
                  <Loader2 className="h-4 w-4 animate-spin" />
                  <AlertDescription>
                    Simulation is in progress. Please wait...
                  </AlertDescription>
                </Alert>
              )}
              {status === "completed" && stats && (
                <div className="space-y-3 text-sm text-gray-800">
                  <p>Total Agents: {stats.total_agents}</p>
                  <p>Escaped: {stats.escaped}</p>
                  <p>Dead: {stats.dead}</p>
                  <p>Followers: {stats.followers}</p>
                  <p>No Repulsion: {stats.no_repulsion}</p>
                  <p>Exit Avoiders: {stats.avoiders}</p>

                  <Button
                    onClick={downloadSimulation}
                    variant="outline"
                    className="mt-4 w-full border-red-600 text-red-600 hover:bg-red-50"
                  >
                    <Download className="mr-2 h-4 w-4" />
                    Download Simulation MP4
                  </Button>
                </div>
              )}
              {status === "error" && error && (
                <Alert variant="destructive">
                  <AlertDescription>{error}</AlertDescription>
                </Alert>
              )}
            </CardContent>
          </Card>
        </div>
      </div>
    </div>
  );
}
