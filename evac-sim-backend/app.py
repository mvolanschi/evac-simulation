from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-GUI backend
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter
import math
import random
from matplotlib.patches import Circle
import io
import base64
import tempfile
import os
from threading import Thread
import uuid
import time

app = Flask(__name__)
CORS(app)

# Store simulation results temporarily
simulation_results = {}

class Agent:
    """
    Agent class representing individuals in the evacuation simulation.
    Each agent has a position, orientation, and various behavioral traits.
    """
    def __init__(self, agent_id, x, y, force_follower=False, force_ignores_repulsion=False, force_avoids_crowded=False):
        """Initialize a new agent with given ID and position"""
        self.id = agent_id
        self.pos = np.array([x, y], dtype=float)  # Current position
        self.target = None  # Target exit (will be set during simulation)
        self.reached = False  # Whether agent has reached an exit
        self.orientation = 0.0  # Direction agent is facing (radians)
        self.received_positions = {}  # Dictionary to store positions of other agents
        self.mode = "search"  # Initial mode: searching for exits
        self.follower_id = None  # ID of agent this agent is following (if any)
        self.panic_speed = random.uniform(0.5, 1.2) * 2  # Random speed multiplier in panic mode
        self.alive = True  # Whether agent is alive (not caught by fire)
        
        # Personality traits - determine behavior
        self.is_follower = force_follower
        self.ignores_repulsion = force_ignores_repulsion
        self.avoids_crowded_exits = force_avoids_crowded
        
    def receive_positions(self, agents):
        """
        Update the agent's knowledge of other agents' positions
        Only includes alive agents (not the agent itself)
        """
        self.received_positions = {a.id: a.pos.copy() for a in agents if a.id != self.id and a.alive}

    def compute_potential(self, pos, params):
        """
        Compute the potential field value at a given position.
        Lower potential is better (agent will move "downhill" in potential field).
        Includes attraction to target and repulsion from obstacles and other agents.
        """
        # If no target, return zero potential (no gradient)
        if self.target is None:
            return 0

        # Compute distance to target for attraction component
        dx, dy = pos - self.target
        dist = np.hypot(dx, dy)
        U = params['ATTR_COEFF'] * dist  # Linear attraction to target

        # Calculate distance to closest exit for repulsion scaling
        exit_dist = np.inf
        if self.target is not None:
            exit_dist = np.linalg.norm(pos - self.target)
        else:
            # If no target, find closest exit
            for ex in params['exits']:
                curr_dist = np.linalg.norm(pos - ex)
                if curr_dist < exit_dist:
                    exit_dist = curr_dist

        # Gradually reduce repulsion as agent approaches exit
        # Full strength at distance 15, zero at distance 5
        fade_factor = np.clip((exit_dist - 5) / 10, 0.0, 1.0)

        # Skip repulsion calculation if this agent ignores repulsion
        if self.ignores_repulsion:
            return U
            
        # Set repulsion coefficients based on panic mode
        long_rep = (params['PANIC_REP_COEFF'] if params['PANIC_MODE'] else params['REP_COEFF']) * fade_factor
        short_rep = (params['PANIC_REP_SHORT_COEFF'] if params['PANIC_MODE'] else params['REP_SHORT_COEFF']) * fade_factor

        # Repulsion from obstacles
        for obs in params['obstacles']:
            d = np.linalg.norm(pos - obs)
            if d < params['MAX_REP_RADIUS']:
                U += long_rep * (1.0 / d**2 - 1.0 / params['MAX_REP_RADIUS']**2)
            if d < params['SHORT_REP_RADIUS']:
                U += short_rep * (1.0 / d**2 - 1.0 / params['SHORT_REP_RADIUS']**2)

        # Repulsion from other agents
        for pos_other in self.received_positions.values():
            d = np.linalg.norm(pos - pos_other)
            if d < params['MAX_REP_RADIUS']:
                U += long_rep * (1.0 / d - 1.0 / params['MAX_REP_RADIUS'])
            if d < params['SHORT_REP_RADIUS']:
                U += short_rep * (1.0 / d - 1.0 / params['SHORT_REP_RADIUS'])

        return U

    def compute_gradient(self, params, delta=1e-3):
        """
        Compute the gradient of the potential field at the agent's position.
        Used to determine the direction of movement.
        """
        pos = self.pos
        # Compute potential at slightly offset positions
        Ux1 = self.compute_potential(pos + np.array([delta, 0]), params)
        Ux2 = self.compute_potential(pos - np.array([delta, 0]), params)
        Uy1 = self.compute_potential(pos + np.array([0, delta]), params)
        Uy2 = self.compute_potential(pos - np.array([0, delta]), params)
        
        # Finite difference approximation of gradient
        grad = np.array([(Ux1 - Ux2) / (2 * delta), (Uy1 - Uy2) / (2 * delta)])
        return grad
        
    def find_exit(self, agents, params):
        """
        Find an appropriate exit for the agent to target.
        Takes into account if the agent avoids crowded exits.
        """
        best_exit = None
        min_dist = float('inf')
        
        for ex in params['exits']:
            dist = np.linalg.norm(self.pos - ex)
            
            if dist < 10:  # Exit detection radius
                # Count how many agents are near this exit
                nearby_agents = sum(1 for a in agents if a.alive and not a.reached and np.linalg.norm(a.pos - ex) < 5)
                
                # If agent avoids crowded exits and this exit is crowded, skip it
                if self.avoids_crowded_exits and nearby_agents > params['CROWDED_THRESHOLD']:
                    continue
                    
                # Choose closest exit
                if dist < min_dist:
                    min_dist = dist
                    best_exit = ex.copy()
        
        return best_exit

    def update(self, agents, params):
        """
        Update agent position and state for the next time step.
        This is the main behavior method for each agent.
        """
        # Skip update if agent is dead or has already escaped
        if not self.alive or self.reached:
            return
            
        # Check if agent has reached the target exit
        if self.target is not None and np.linalg.norm(self.pos - self.target) < params['ESCAPE_DISTANCE']:
            self.reached = True
            return

        # === FOLLOWER BEHAVIOR ===
        # If following another agent, move toward that agent
        if self.follower_id is not None:
            leader = next((a for a in agents if a.id == self.follower_id and a.alive), None)
            if leader:
                direction = leader.pos - self.pos
                norm = np.linalg.norm(direction)
                if norm > 1e-5:
                    direction /= norm
                    self.pos += direction * params['BASE_STEP_SIZE']
                    self.orientation = math.atan2(direction[1], direction[0])
                return
            else:
                # If leader is gone (dead or escaped), stop following
                self.follower_id = None

        # === SEARCH BEHAVIOR ===
        # If agent doesn't have a target exit, search for one
        if self.target is None:
            self.mode = "search"
            
            # Try to find an appropriate exit
            best_exit = self.find_exit(agents, params)
            if best_exit is not None:
                self.target = best_exit
                self.mode = "escape"
            
            # If still searching (no exit found), move randomly
            if self.mode == "search":
                angle = random.uniform(0, 2 * np.pi)
                direction = np.array([np.cos(angle), np.sin(angle)])
                self.pos += direction * params['BASE_STEP_SIZE'] * 0.5  # Move slower while searching
                self.orientation = math.atan2(direction[1], direction[0])
                return

        # === ESCAPE BEHAVIOR ===
        # Move toward target using potential field gradient
        grad = self.compute_gradient(params)
        velocity = -grad  # Move "downhill" in potential field
        
        # Normalize velocity if it's non-zero
        norm = np.linalg.norm(velocity)
        if norm > 1e-5:
            velocity /= norm
            
            # Add small random noise to prevent deadlocks
            noise = params['NOISE_AMPLITUDE'] * np.random.randn()
            perp = np.array([-velocity[1], velocity[0]])  # Perpendicular direction
            velocity += noise * perp
            velocity /= np.linalg.norm(velocity)  # Renormalize
            
            # Calculate speed based on panic mode
            speed = params['BASE_STEP_SIZE'] * (self.panic_speed if params['PANIC_MODE'] else 1.0)
            
            # Increase speed when near exit to help push through congestion
            exit_dist = np.linalg.norm(self.pos - self.target)
            if exit_dist < 5:
                speed *= 1.2  # Speed boost near exits
                
            # Update position and orientation
            self.pos += velocity * speed
            self.orientation = math.atan2(velocity[1], velocity[0])

def create_simulation_params(config):
    """Create simulation parameters from config"""
    GRID_ROWS, GRID_COLS = config['grid_rows'], config['grid_cols']
    ROOM_WIDTH = GRID_COLS * 1.0
    ROOM_HEIGHT = GRID_ROWS * 1.0
    
    exits = [
        np.array([0.5, GRID_ROWS/2]), 
        np.array([GRID_COLS-0.5, GRID_ROWS/2]),
        np.array([GRID_COLS/2, 0.5]), 
        np.array([GRID_COLS/2, GRID_ROWS-0.5])
    ]
    
    # Create obstacles - now configurable
    obstacles = []
    NUM_OBSTACLES = config.get('num_obstacles', 15)  # Use config value or default to 15
    OBSTACLE_MIN_DIST = 2
    
    while len(obstacles) < NUM_OBSTACLES:
        x = random.uniform(OBSTACLE_MIN_DIST, ROOM_WIDTH - OBSTACLE_MIN_DIST)
        y = random.uniform(OBSTACLE_MIN_DIST, ROOM_HEIGHT - OBSTACLE_MIN_DIST)
        new_obstacle = np.array([x, y])
        
        if all(np.linalg.norm(new_obstacle - exit_pos) > 3 for exit_pos in exits):
            obstacles.append(new_obstacle)
    
    return {
        'GRID_ROWS': GRID_ROWS,
        'GRID_COLS': GRID_COLS,
        'ROOM_WIDTH': ROOM_WIDTH,
        'ROOM_HEIGHT': ROOM_HEIGHT,
        'exits': exits,
        'obstacles': obstacles,
        'PANIC_MODE': True,
        'NOISE_AMPLITUDE': 1e-3,
        'ATTR_COEFF': 0.4,
        'REP_COEFF': 1.5,
        'REP_SHORT_COEFF': 0.7,
        'PANIC_REP_COEFF': 2.5,
        'PANIC_REP_SHORT_COEFF': 3.0,
        'MAX_REP_RADIUS': 2.7,
        'SHORT_REP_RADIUS': 0.6,
        'BASE_STEP_SIZE': config.get('agent_speed', 0.015),  # Use configurable agent speed
        'FIRE_GROWTH_RATE': config['fire_growth_rate'],
        'ESCAPE_DISTANCE': 1.0,
        'CROWDED_THRESHOLD': 5
    }

def run_simulation(config, simulation_id):
    """Run the evacuation simulation with given config"""
    try:
        params = create_simulation_params(config)
        
        # Create agents
        agents = []
        NUM_AGENTS = config['num_agents']
        
        # Calculate number of agents with each trait
        num_followers = int(NUM_AGENTS * config['follower_percentage'] / 100)
        num_no_repulsion = int(NUM_AGENTS * config['no_repulsion_percentage'] / 100)
        num_avoiders = int(NUM_AGENTS * config['crowded_exit_avoider_percentage'] / 100)
        
        for i in range(NUM_AGENTS):
            while True:
                x, y = np.random.uniform(2, params['GRID_COLS']-2), np.random.uniform(2, params['GRID_ROWS']-2)
                if all(np.linalg.norm(np.array([x, y]) - obs) > 1 for obs in params['obstacles']):
                    force_follower = i < num_followers
                    force_ignores_repulsion = force_follower or (num_followers <= i < (num_followers + num_no_repulsion))
                    force_avoids_crowded = (num_followers + num_no_repulsion) <= i < (num_followers + num_no_repulsion + num_avoiders)
                    
                    agents.append(Agent(i, x, y, 
                                       force_follower=force_follower,
                                       force_ignores_repulsion=force_ignores_repulsion, 
                                       force_avoids_crowded=force_avoids_crowded))
                    break

        # Initialize fire
        fire_center = np.array([random.uniform(5, params['GRID_COLS']-5), random.uniform(5, params['GRID_ROWS']-5)])
        fire_radius = 0
        fire_active = False
        
        # Create visualization
        fig, ax = plt.subplots(figsize=(12, 10))
        ax.set_xlim(0, params['GRID_COLS'])
        ax.set_ylim(0, params['GRID_ROWS'])
        ax.set_aspect('equal')
        
        # Plot elements
        active_agents_plot, = ax.plot([], [], 'ro', label='Active', markersize=4)
        escaped_agents_plot, = ax.plot([], [], 'go', label='Escaped', markersize=4)
        dead_agents_plot, = ax.plot([], [], 'ko', label='Dead', markersize=4)
        follower_agents_plot, = ax.plot([], [], 'bo', label='Following', markersize=4)
        no_rep_agents_plot, = ax.plot([], [], 'yo', label='No Repulsion', markersize=4)
        avoider_agents_plot, = ax.plot([], [], 'mo', label='Exit Avoiders', markersize=4)
        
        obstacles_plot, = ax.plot(*zip(*params['obstacles']) if params['obstacles'] else ([],[]), 'ks', markersize=8, label='Obstacles')
        exits_plot, = ax.plot(*zip(*params['exits']), 'gs', markersize=12, label='Exits')
        
        fire_patch = Circle(fire_center, fire_radius, color='red', alpha=0.5)
        fire_circle = ax.add_patch(fire_patch)
        
        # Create orientation lines for each agent
        orients = [ax.plot([], [], 'r-', linewidth=1)[0] for _ in agents]
        
        # Create text elements for statistics display
        time_text = ax.text(0.02, 0.98, '', transform=ax.transAxes, fontsize=10, verticalalignment='top')
        escaped_text = ax.text(0.02, 0.94, '', transform=ax.transAxes, fontsize=10, verticalalignment='top')
        active_text = ax.text(0.02, 0.90, '', transform=ax.transAxes, fontsize=10, verticalalignment='top')
        dead_text = ax.text(0.02, 0.86, '', transform=ax.transAxes, fontsize=10, verticalalignment='top')
        
        ax.legend(loc='upper right', fontsize=8)
        ax.set_title('Evacuation Simulation')
        ax.grid(True, alpha=0.3)
        
        def update_frame(frame):
            nonlocal fire_radius, fire_active
            
            # Fire logic
            if frame >= 50:  # Start fire after ~2.5 seconds (50 frames at 20fps)
                fire_active = True
            
            if fire_active:
                fire_radius += params['FIRE_GROWTH_RATE']
                fire_patch.set_radius(fire_radius)
            
            # Agent updates
            for agent in agents:
                agent.receive_positions(agents)
            
            # Collision and following logic
            for i, a1 in enumerate(agents):
                for j, a2 in enumerate(agents[i+1:], i+1):
                    if a1.alive and a2.alive and a1.follower_id is None and a2.follower_id is None:
                        dist = np.linalg.norm(a1.pos - a2.pos)
                        if dist < 0.4:
                            if a1.is_follower and random.random() < 0.7:
                                a1.follower_id = a2.id
                                a1.mode = "follow"
                            elif a2.is_follower and random.random() < 0.7:
                                a2.follower_id = a1.id
                                a2.mode = "follow"
                            elif not a1.ignores_repulsion and not a2.ignores_repulsion:
                                direction = a1.pos - a2.pos
                                if np.linalg.norm(direction) > 0:
                                    normalized = direction / np.linalg.norm(direction)
                                    a1.pos += normalized * 0.1
                                    a2.pos -= normalized * 0.1
            
            # Fire damage
            if fire_active:
                for agent in agents:
                    if agent.alive and not agent.reached:
                        dist_to_fire = np.linalg.norm(agent.pos - fire_center)
                        if dist_to_fire < fire_radius:
                            agent.alive = False
            
            # Move agents
            for agent in agents:
                agent.update(agents, params)
            
            # Update visualization
            active_pos = [a.pos for a in agents if a.alive and not a.reached and a.follower_id is None and not a.ignores_repulsion and not a.avoids_crowded_exits]
            escaped_pos = [a.pos for a in agents if a.reached]
            dead_pos = [a.pos for a in agents if not a.alive]
            follower_pos = [a.pos for a in agents if a.alive and not a.reached and a.follower_id is not None]
            no_rep_pos = [a.pos for a in agents if a.alive and not a.reached and a.ignores_repulsion and a.follower_id is None]
            avoider_pos = [a.pos for a in agents if a.alive and not a.reached and a.avoids_crowded_exits and a.follower_id is None and not a.ignores_repulsion]
            
            active_agents_plot.set_data(
                [p[0] for p in active_pos] if active_pos else [],
                [p[1] for p in active_pos] if active_pos else []
            )
            escaped_agents_plot.set_data(
                [p[0] for p in escaped_pos] if escaped_pos else [],
                [p[1] for p in escaped_pos] if escaped_pos else []
            )
            dead_agents_plot.set_data(
                [p[0] for p in dead_pos] if dead_pos else [],
                [p[1] for p in dead_pos] if dead_pos else []
            )
            follower_agents_plot.set_data(
                [p[0] for p in follower_pos] if follower_pos else [],
                [p[1] for p in follower_pos] if follower_pos else []
            )
            no_rep_agents_plot.set_data(
                [p[0] for p in no_rep_pos] if no_rep_pos else [],
                [p[1] for p in no_rep_pos] if no_rep_pos else []
            )
            avoider_agents_plot.set_data(
                [p[0] for p in avoider_pos] if avoider_pos else [],
                [p[1] for p in avoider_pos] if avoider_pos else []
            )
            
            # Update orientation lines for each agent
            for i, agent in enumerate(agents):
                if agent.alive and not agent.reached:
                    # Draw line showing agent orientation (arrows)
                    dx = 0.4 * np.cos(agent.orientation)
                    dy = 0.4 * np.sin(agent.orientation)
                    orients[i].set_data([agent.pos[0], agent.pos[0] + dx], [agent.pos[1], agent.pos[1] + dy])
                    
                    # Set color based on agent type
                    if agent.follower_id is not None:
                        orients[i].set_color('blue')
                    elif agent.ignores_repulsion:
                        orients[i].set_color('yellow')
                    elif agent.avoids_crowded_exits:
                        orients[i].set_color('magenta')
                    else:
                        orients[i].set_color('red')
                else:
                    # Clear orientation line for dead or escaped agents
                    orients[i].set_data([], [])
            
            # Update statistics text
            time_text.set_text(f'Time: {frame/20:.1f}s')
            escaped_text.set_text(f'Escaped: {sum(a.reached for a in agents)}/{NUM_AGENTS}')
            active_text.set_text(f'Active: {sum(a.alive and not a.reached for a in agents)}/{NUM_AGENTS}')
            dead_text.set_text(f'Dead: {sum(not a.alive for a in agents)}/{NUM_AGENTS}')
            
            return ([active_agents_plot, escaped_agents_plot, dead_agents_plot, 
                    follower_agents_plot, no_rep_agents_plot, avoider_agents_plot, 
                    fire_circle, time_text, escaped_text, active_text, dead_text] + orients)
        
        # Create animation
        ani = FuncAnimation(fig, update_frame, frames=600, interval=50, blit=True)
        
        # Save animation as MP4
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
        
        # Use FFMpegWriter for MP4 output
        writer = FFMpegWriter(fps=20, metadata=dict(artist='Evacuation Simulation'), bitrate=1800)
        ani.save(temp_file.name, writer=writer)
        
        # Store result
        simulation_results[simulation_id] = {
            'status': 'completed',
            'file_path': temp_file.name,
            'stats': {
                'total_agents': NUM_AGENTS,
                'escaped': sum(1 for a in agents if a.reached),
                'dead': sum(1 for a in agents if not a.alive),
                'followers': num_followers,
                'no_repulsion': num_no_repulsion,
                'avoiders': num_avoiders,
                'obstacles': len(params['obstacles']),
                'survival_rate': sum(1 for a in agents if a.reached) / NUM_AGENTS * 100,
                'casualty_rate': sum(1 for a in agents if not a.alive) / NUM_AGENTS * 100,
                'agent_speed': config.get('agent_speed', 0.015)  # Include agent speed in stats
            }
        }
        
        plt.close(fig)
        
    except Exception as e:
        simulation_results[simulation_id] = {
            'status': 'error',
            'error': str(e)
        }

@app.route('/api/simulate', methods=['POST'])
def simulate():
    """Start a new simulation"""
    try:
        config = request.json
        simulation_id = str(uuid.uuid4())
        
        # Validate config
        required_fields = ['grid_rows', 'grid_cols', 'num_agents', 'fire_growth_rate', 
                          'follower_percentage', 'no_repulsion_percentage', 'crowded_exit_avoider_percentage']
        
        for field in required_fields:
            if field not in config:
                return jsonify({'error': f'Missing field: {field}'}), 400
        
        # Validate ranges
        if not (5 <= config['grid_rows'] <= 50) or not (5 <= config['grid_cols'] <= 50):
            return jsonify({'error': 'Grid dimensions must be between 5 and 50'}), 400
        
        if not (10 <= config['num_agents'] <= 200):
            return jsonify({'error': 'Number of agents must be between 10 and 200'}), 400
        
        if not (0 <= config['fire_growth_rate'] <= 1):
            return jsonify({'error': 'Fire growth rate must be between 0 and 1'}), 400
        
        # Validate obstacle count if provided
        if 'num_obstacles' in config:
            if not (0 <= config['num_obstacles'] <= 50):
                return jsonify({'error': 'Number of obstacles must be between 0 and 50'}), 400
        
        # Validate agent speed if provided
        if 'agent_speed' in config:
            if not (0.005 <= config['agent_speed'] <= 0.1):
                return jsonify({'error': 'Agent speed must be between 0.005 and 0.1'}), 400
        
        for percentage_field in ['follower_percentage', 'no_repulsion_percentage', 'crowded_exit_avoider_percentage']:
            if not (0 <= config[percentage_field] <= 100):
                return jsonify({'error': f'{percentage_field} must be between 0 and 100'}), 400
        
        # Start simulation in background thread
        simulation_results[simulation_id] = {'status': 'running'}
        thread = Thread(target=run_simulation, args=(config, simulation_id))
        thread.daemon = True
        thread.start()
        
        return jsonify({'simulation_id': simulation_id})
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/status/<simulation_id>')
def get_status(simulation_id):
    """Get simulation status"""
    if simulation_id not in simulation_results:
        return jsonify({'error': 'Simulation not found'}), 404
    
    result = simulation_results[simulation_id]
    if result['status'] == 'completed':
        return jsonify({
            'status': 'completed',
            'stats': result['stats']
        })
    elif result['status'] == 'error':
        return jsonify({
            'status': 'error',
            'error': result['error']
        })
    else:
        return jsonify({'status': result['status']})

@app.route('/api/download/<simulation_id>')
def download_simulation(simulation_id):
    """Download simulation MP4"""
    if simulation_id not in simulation_results:
        return jsonify({'error': 'Simulation not found'}), 404
    
    result = simulation_results[simulation_id]
    if result['status'] != 'completed':
        return jsonify({'error': 'Simulation not completed'}), 400
    
    try:
        return send_file(result['file_path'], as_attachment=True, download_name='evacuation_simulation.mp4')
    except Exception as e:
        return jsonify({'error': f'Failed to send file: {str(e)}'}), 500

@app.route('/api/cleanup/<simulation_id>', methods=['DELETE'])
def cleanup_simulation(simulation_id):
    """Clean up simulation files and data"""
    if simulation_id not in simulation_results:
        return jsonify({'error': 'Simulation not found'}), 404
    
    result = simulation_results[simulation_id]
    
    # Clean up temporary file if it exists
    if result['status'] == 'completed' and 'file_path' in result:
        try:
            if os.path.exists(result['file_path']):
                os.unlink(result['file_path'])
        except Exception as e:
            print(f"Warning: Failed to cleanup file {result['file_path']}: {e}")
    
    # Remove from results
    del simulation_results[simulation_id]
    
    return jsonify({'message': 'Simulation cleaned up successfully'})

@app.route('/health')
def health():
    return jsonify({'status': 'healthy', 'simulations_count': len(simulation_results)})

@app.route('/api/simulations')
def list_simulations():
    """List all current simulations"""
    return jsonify({
        'simulations': {
            sim_id: {'status': result['status']} 
            for sim_id, result in simulation_results.items()
        }
    })

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=int(os.environ.get('PORT', 5001)))