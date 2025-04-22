import torch, numpy as np
from agent import Agent
from datetime import datetime
from models import MlpMinigridPolicy
from replay_buffer import ReplayBuffer
from environment import MinigridDoorKey6x6ImgObs
from utils import set_seed, soft_update, save_as_onnx
from torch.nn import SmoothL1Loss

def run_episode(env, agent, seed=None):
    state = env.reset(seed=seed)[0]
    score = 0
    done = False
    while not done:
        action = agent.select_action(state)
        state, reward, terminated, truncated, _ = env.step(action)
        score += reward
        done = terminated or truncated
    return score

if __name__ == "__main__":
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    num_episodes = 1000
    buffer_size = 100000
    epsilon_start = 1.0
    epsilon_end = 0.05
    epsilon_decay = 5000
    gamma = 0.99
    learning_rate = 0.0001
    tau = 1e-3
    update_after = 2000
    minibatch_size = 128
    eval_episodes = 50
    save_interval = 50
    seed = 0

    set_seed(seed)

    env = MinigridDoorKey6x6ImgObs()
    print("Using DoorKey-6x6 environment")

    num_actions = env.action_space.n
    state_space = env.observation_space.shape
    print(f"Action space: {num_actions}, State space: {state_space}")

    timestamp = datetime.now().strftime('%Y_%m_%d-%H_%M_%S')
    save_path = f"minigrid-model-{timestamp}.pt"
    onnx_path = f"submission_model.onnx"
    print(f"Model will be saved to: {save_path}")

    dqn = MlpMinigridPolicy(num_actions=num_actions).to(device=device)
    dqn_target = MlpMinigridPolicy(num_actions=num_actions).to(device=device)
    dqn_target.load_state_dict(dqn.state_dict())
    
    optimizer = torch.optim.Adam(dqn.parameters(), lr=learning_rate, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=250, gamma=0.5)
    criterion = SmoothL1Loss()
    buffer = ReplayBuffer(num_actions=num_actions, memory_len=buffer_size)
    
    returns = []
    losses = []
    timesteps = 0
    
    state = env.reset()[0]
    
    for i in range(num_episodes):
        ret = 0
        done = False
        
        while not done:
            epsilon = epsilon_end + (epsilon_start - epsilon_end) * np.exp(-1.0 * timesteps / epsilon_decay)
            
            if np.random.random() < epsilon:
                a = np.random.randint(low=0, high=num_actions)
            else:
                state_tensor = torch.tensor(state, dtype=torch.float32, device=device)
                net_out = dqn(state_tensor).detach().cpu().numpy()
                a = np.argmax(net_out)
            
            next_state, r, terminated, truncated, _ = env.step(a)
            done = terminated or truncated
            ret += r
            
            buffer.add(state, a, r, next_state, done)
            
            state = next_state
            timesteps += 1
            
            if buffer.length() > minibatch_size and buffer.length() > update_after:
                optimizer.zero_grad()
                
                states_mb, mb_a, mb_reward, next_states_mb, mb_done = buffer.sample_batch(batch_size=minibatch_size, device=device)
                
                states_tensor = torch.tensor(states_mb, dtype=torch.float32, device=device)
                next_states_tensor = torch.tensor(next_states_mb, dtype=torch.float32, device=device)
                
                q_values = dqn(states_tensor)
                current_q = torch.sum(q_values * mb_a, dim=1)
                
                with torch.no_grad():
                    next_actions = dqn(next_states_tensor).argmax(dim=1, keepdim=True)
                    next_q_values = dqn_target(next_states_tensor)
                    next_q_values = next_q_values.gather(1, next_actions).squeeze()
                    target_q = mb_reward + (1 - mb_done) * gamma * next_q_values
                
                td_errors = (target_q - current_q).detach().cpu().numpy()
                
                sample_indices = np.random.choice(buffer.length(), size=min(minibatch_size, buffer.length()), replace=False)
                
                buffer.update_priorities(sample_indices, td_errors)
                
                loss = criterion(current_q, target_q)
                loss.backward()
                
                torch.nn.utils.clip_grad_norm_(dqn.parameters(), max_norm=10.0)
                
                optimizer.step()
                losses.append(loss.item())
                
                soft_update(dqn, dqn_target, tau)
            
            if done:
                state = env.reset()[0]
                print(f"Episode: {i}\tReturn: {ret}\tTimestep: {timesteps}\tEpsilon: {epsilon:.4f}")
                break
        
        returns.append(ret)
        
        scheduler.step()
        
        if (i + 1) % save_interval == 0:
            checkpoint_dict = {'model_params': dqn.state_dict(), 'timesteps': timesteps}
            torch.save(checkpoint_dict, save_path)
            print(f'Saved checkpoint to {save_path}')
            
            avg_return = sum(returns[-50:]) / min(len(returns), 50)
            print(f'Episode {i+1}/{num_episodes}\tAverage Return (last 50): {avg_return:.2f}')
    
    agent = Agent(model=dqn, device=device)
    scores = []
    for i in range(eval_episodes):
        seed = np.random.randint(1e7) if seed is None else np.random.randint(1e7)
        eval_env = MinigridDoorKey6x6ImgObs()
        score = run_episode(eval_env, agent, seed=seed)
        scores.append(score)
        eval_env.close()
    
    print(f"Final Evaluation: Average Return = {np.mean(scores):.4f}")
    
    final_checkpoint = {'model_params': dqn.state_dict(), 'timesteps': timesteps}
    torch.save(final_checkpoint, save_path)
    
    sample_state = env.reset()[0]
    sample_input = torch.tensor(sample_state, dtype=torch.float32, device=device)
    save_as_onnx(dqn, sample_input, onnx_path)
    print(f"Model saved to {onnx_path}")