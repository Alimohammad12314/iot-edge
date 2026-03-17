import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# --- PAGE CONFIG ---
st.set_page_config(page_title="IoT Edge Resource Allocation", layout="wide")

# --- TITLES & INTRODUCTION ---
st.title("🌐 Explainable Resource Allocation Framework for IoT Edge Computing")
st.markdown("""
**The Challenge:** Edge devices in IoT networks have limited CPU and memory. While Machine Learning (ML) can optimize how tasks are distributed among these devices, traditional ML acts as a "black box," lacking the transparency needed for critical systems.  
**Our Solution:** A dynamic framework that uses a Random Forest classifier to intelligently route tasks between *Best-Fit Resource-Aware (BFRA)* and *First-Fit Round-Robin (FFRR)* algorithms, complete with explainable AI (XAI) features to build trust and accountability.
""")
st.divider()

# --- SIDEBAR CREDITS ---
st.sidebar.markdown("### Project Details")
st.sidebar.markdown("**Title:** Explainable Resource Allocation Framework for IoT Edge Computing")
st.sidebar.markdown("**Presented by:**")
st.sidebar.markdown("- Maimoona (22COB131)")
st.sidebar.markdown("- Mohammad Ali (22COB229)")
st.sidebar.markdown("**Supervisor:**")
st.sidebar.markdown("- Prof. M. M. Sufyan Beg")
st.sidebar.divider()

# --- CACHED DATA & TRAINING (Runs Once) ---
@st.cache_resource
def train_model():
    np.random.seed(42)
    def generate_task_stream_local(num_tasks, arrival_interval, cpu_range, mem_range, delay_range):
        tasks = []
        time = 0.0
        for i in range(num_tasks):
            cpu_req = np.random.randint(cpu_range[0], cpu_range[1]+1)
            mem_req = np.random.randint(mem_range[0], mem_range[1]+1)
            max_delay = np.random.randint(delay_range[0], delay_range[1]+1)
            tasks.append({'id': i, 'arrival': time, 'cpu': cpu_req, 'mem': mem_req, 'max_delay': max_delay, 'duration': cpu_req})
            time += arrival_interval
        return tasks

    def sim_for_training(tasks, nodes_spec, strategy):
        num_nodes = len(nodes_spec)
        free_cpu = [node['cpu'] for node in nodes_spec]
        free_mem = [node['mem'] for node in nodes_spec]
        running = [[] for _ in range(num_nodes)]
        pointer = 0; queue = []; finish_times = {}; free_cpu_hist = []; queue_len_hist = []

        for task in sorted(tasks, key=lambda t: t['arrival']):
            current_time = task['arrival']
            for i in range(num_nodes):
                still_running = []
                for r in running[i]:
                    if r['end_time'] <= current_time:
                        free_cpu[i] += r['cpu']; free_mem[i] += r['mem']
                    else: still_running.append(r)
                running[i] = still_running
            
            free_cpu_hist.append(list(free_cpu)); queue_len_hist.append(len(queue))

            if strategy == 'BFRA':
                best_idx, best_left = None, None
                for i in range(num_nodes):
                    if free_cpu[i] >= task['cpu'] and free_mem[i] >= task['mem']:
                        slack = free_cpu[i] - task['cpu']
                        if best_idx is None or slack < best_left: best_idx, best_left = i, slack
                if best_idx is not None:
                    free_cpu[best_idx] -= task['cpu']; free_mem[best_idx] -= task['mem']
                    end_t = current_time + task['duration']
                    running[best_idx].append({'id': task['id'], 'end_time': end_t, 'cpu': task['cpu'], 'mem': task['mem']})
                    finish_times[task['id']] = end_t
                else: queue.append(task)
            else:
                assigned = False
                for offset in range(num_nodes):
                    i = (pointer + offset) % num_nodes
                    if free_cpu[i] >= task['cpu'] and free_mem[i] >= task['mem']:
                        free_cpu[i] -= task['cpu']; free_mem[i] -= task['mem']
                        end_t = current_time + task['duration']
                        running[i].append({'id': task['id'], 'end_time': end_t, 'cpu': task['cpu'], 'mem': task['mem']})
                        finish_times[task['id']] = end_t
                        pointer = (i + 1) % num_nodes
                        assigned = True; break
                if not assigned: queue.append(task)

            # Process Queue
            scheduled = True
            while scheduled and queue:
                scheduled = False
                if strategy == 'BFRA':
                    for qtask in queue[:]:
                        best_idx, best_left = None, None
                        for i in range(num_nodes):
                            if free_cpu[i] >= qtask['cpu'] and free_mem[i] >= qtask['mem']:
                                slack = free_cpu[i] - qtask['cpu']
                                if best_idx is None or slack < best_left: best_idx, best_left = i, slack
                        if best_idx is not None:
                            free_cpu[best_idx] -= qtask['cpu']; free_mem[best_idx] -= qtask['mem']
                            end_t = current_time + qtask['duration']
                            running[best_idx].append({'id': qtask['id'], 'end_time': end_t, 'cpu': qtask['cpu'], 'mem': qtask['mem']})
                            finish_times[qtask['id']] = end_t; queue.remove(qtask); scheduled = True
                else:
                    for qtask in queue[:]:
                        for offset in range(num_nodes):
                            i = (pointer + offset) % num_nodes
                            if free_cpu[i] >= qtask['cpu'] and free_mem[i] >= qtask['mem']:
                                free_cpu[i] -= qtask['cpu']; free_mem[i] -= qtask['mem']
                                end_t = current_time + qtask['duration']
                                running[i].append({'id': qtask['id'], 'end_time': end_t, 'cpu': qtask['cpu'], 'mem': qtask['mem']})
                                finish_times[qtask['id']] = end_t; queue.remove(qtask); pointer = (i + 1) % num_nodes; scheduled = True; break

        # Final Drain
        while queue:
            next_finish, next_node = None, None
            for i in range(num_nodes):
                for r in running[i]:
                    if next_finish is None or r['end_time'] < next_finish:
                        next_finish = r['end_time']; next_node = i
            if next_finish is None: break
            current_time = next_finish
            remaining = []
            for r in running[next_node]:
                if r['end_time'] == next_finish:
                    free_cpu[next_node] += r['cpu']; free_mem[next_node] += r['mem']
                else: remaining.append(r)
            running[next_node] = remaining

            scheduled = True
            while scheduled and queue:
                scheduled = False
                if strategy == 'BFRA':
                    for qtask in queue[:]:
                        best_idx, best_left = None, None
                        for i in range(num_nodes):
                            if free_cpu[i] >= qtask['cpu'] and free_mem[i] >= qtask['mem']:
                                slack = free_cpu[i] - qtask['cpu']
                                if best_idx is None or slack < best_left: best_idx, best_left = i, slack
                        if best_idx is not None:
                            free_cpu[best_idx] -= qtask['cpu']; free_mem[best_idx] -= qtask['mem']
                            end_t = current_time + qtask['duration']
                            running[best_idx].append({'id': qtask['id'], 'end_time': end_t, 'cpu': qtask['cpu'], 'mem': qtask['mem']})
                            finish_times[qtask['id']] = end_t; queue.remove(qtask); scheduled = True
                else:
                    for qtask in queue[:]:
                        for offset in range(num_nodes):
                            i = (pointer + offset) % num_nodes
                            if free_cpu[i] >= qtask['cpu'] and free_mem[i] >= qtask['mem']:
                                free_cpu[i] -= qtask['cpu']; free_mem[i] -= qtask['mem']
                                end_t = current_time + qtask['duration']
                                running[i].append({'id': qtask['id'], 'end_time': end_t, 'cpu': qtask['cpu'], 'mem': qtask['mem']})
                                finish_times[qtask['id']] = end_t; queue.remove(qtask); pointer = (i + 1) % num_nodes; scheduled = True; break
        return finish_times, free_cpu_hist, queue_len_hist

    nodes = [{'cpu':100, 'mem':1000} for _ in range(3)]
    total_cpu_capacity = 300
    data_rows = []
    
    # Generate 5000 instances (100 episodes * 50 tasks) as per project specs
    for ep in range(100):
        tasks = generate_task_stream_local(50, 1.0, (10, 50), (50, 200), (5, 20))
        bf_finish, bf_free_cpu, bf_queue = sim_for_training(tasks, nodes, 'BFRA')
        ff_finish, _, _ = sim_for_training(tasks, nodes, 'FFRR')
        
        for t, free_cpu_state, queue_len in zip(tasks, bf_free_cpu, bf_queue):
            t_id = t['id']
            # Score = Completion Time + System Congestion Penalty
            score_bf = (bf_finish.get(t_id, np.inf) - t['arrival']) + ((total_cpu_capacity - sum(free_cpu_state)) / total_cpu_capacity)
            score_ff = (ff_finish.get(t_id, np.inf) - t['arrival']) + ((total_cpu_capacity - sum(free_cpu_state)) / total_cpu_capacity)
            label = 0 if score_bf <= score_ff else 1

            row = {'cpu_req': t['cpu'], 'mem_req': t['mem'], 'max_delay': t['max_delay']}
            for i, fc in enumerate(free_cpu_state): row[f'free_cpu_node{i}'] = fc
            row['queue_len'] = queue_len
            row['total_cpu'] = total_cpu_capacity
            row['label'] = label
            data_rows.append(row)

    df = pd.DataFrame(data_rows)
    f_cols = [c for c in df.columns if c != 'label']
    X, y = df[f_cols], df['label']
    # RF Model with 100 trees and balanced weights to handle FFRR imbalance
    clf = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
    clf.fit(X.values, y.values)
    return clf, f_cols, df

clf, feature_cols, train_df = train_model()


# --- SIMULATION LOGIC ---
def generate_task_stream(num_tasks, arrival_interval, cpu_range):
    np.random.seed(np.random.randint(0, 10000))
    tasks = []
    time = 0.0
    for i in range(num_tasks):
        cpu_req = np.random.randint(cpu_range[0], cpu_range[1]+1)
        tasks.append({'id': i, 'arrival': time, 'cpu': cpu_req, 'mem': 100, 'max_delay': 10, 'duration': cpu_req})
        time += arrival_interval
    return tasks

def simulate_with_logs(tasks, nodes_spec, strategy, is_ml=False, clf=None, feature_cols=None):
    num_nodes = len(nodes_spec)
    free_cpu = [node['cpu'] for node in nodes_spec]
    free_mem = [node['mem'] for node in nodes_spec]
    running = [[] for _ in range(num_nodes)]
    pointer = 0; queue = []; finish_times = {}
    total_cpu_capacity = sum(node['cpu'] for node in nodes_spec)
    execution_log = []; queue_history = []

    def assign_task(node_idx, task_dict, curr_time):
        free_cpu[node_idx] -= task_dict['cpu']
        free_mem[node_idx] -= task_dict['mem']
        end_t = curr_time + task_dict['duration']
        running[node_idx].append({'id': task_dict['id'], 'end_time': end_t, 'cpu': task_dict['cpu'], 'mem': task_dict['mem']})
        finish_times[task_dict['id']] = end_t
        execution_log.append({
            'Task': f"Task {task_dict['id']}", 'Node': f"Node {node_idx}", 
            'Start': curr_time, 'Duration': task_dict['duration'], 'Finish': end_t
        })
        return end_t

    for task in sorted(tasks, key=lambda t: t['arrival']):
        current_time = task['arrival']
        queue_history.append({'Time': current_time, 'Queue Length': len(queue)})

        for i in range(num_nodes):
            still_running = []
            for r in running[i]:
                if r['end_time'] <= current_time: free_cpu[i] += r['cpu']; free_mem[i] += r['mem']
                else: still_running.append(r)
            running[i] = still_running

        active_strat = strategy
        if is_ml:
            row_dict = {'cpu_req': task['cpu'], 'mem_req': task['mem'], 'max_delay': task['max_delay'], 'queue_len': len(queue), 'total_cpu': total_cpu_capacity}
            for i, fc in enumerate(free_cpu): row_dict[f'free_cpu_node{i}'] = fc
            X_in = [[row_dict[col] for col in feature_cols]]
            active_strat = 'BFRA' if clf.predict(X_in)[0] == 0 else 'FFRR'

        if active_strat == 'BFRA':
            best_idx, best_left = None, None
            for i in range(num_nodes):
                if free_cpu[i] >= task['cpu'] and free_mem[i] >= task['mem']:
                    slack = free_cpu[i] - task['cpu']
                    if best_idx is None or slack < best_left: best_idx, best_left = i, slack
            if best_idx is not None: assign_task(best_idx, task, current_time)
            else: queue.append(task)
        else:
            assigned = False
            for offset in range(num_nodes):
                i = (pointer + offset) % num_nodes
                if free_cpu[i] >= task['cpu'] and free_mem[i] >= task['mem']:
                    assign_task(i, task, current_time); pointer = (i + 1) % num_nodes; assigned = True; break
            if not assigned: queue.append(task)

        scheduled = True
        while scheduled and queue:
            scheduled = False
            for qtask in queue[:]:
                act_strat_q = strategy
                if is_ml:
                    row_dict = {'cpu_req': qtask['cpu'], 'mem_req': qtask['mem'], 'max_delay': qtask['max_delay'], 'queue_len': len(queue), 'total_cpu': total_cpu_capacity}
                    for i, fc in enumerate(free_cpu): row_dict[f'free_cpu_node{i}'] = fc
                    X_in = [[row_dict[col] for col in feature_cols]]
                    act_strat_q = 'BFRA' if clf.predict(X_in)[0] == 0 else 'FFRR'
                
                assigned = False
                if act_strat_q == 'BFRA':
                    best_idx, best_left = None, None
                    for i in range(num_nodes):
                        if free_cpu[i] >= qtask['cpu'] and free_mem[i] >= qtask['mem']:
                            slack = free_cpu[i] - qtask['cpu']
                            if best_idx is None or slack < best_left: best_idx, best_left = i, slack
                    if best_idx is not None: assign_task(best_idx, qtask, current_time); queue.remove(qtask); assigned = True
                else:
                    for offset in range(num_nodes):
                        i = (pointer + offset) % num_nodes
                        if free_cpu[i] >= qtask['cpu'] and free_mem[i] >= qtask['mem']:
                            assign_task(i, qtask, current_time); queue.remove(qtask); pointer = (i + 1) % num_nodes; assigned = True; break
                if assigned: scheduled = True

    while queue:
        next_finish, next_node = None, None
        for i in range(num_nodes):
            for r in running[i]:
                if next_finish is None or r['end_time'] < next_finish: next_finish = r['end_time']; next_node = i
        if next_finish is None: break
        current_time = next_finish
        queue_history.append({'Time': current_time, 'Queue Length': len(queue)})

        remaining = []
        for r in running[next_node]:
            if r['end_time'] == next_finish: free_cpu[next_node] += r['cpu']; free_mem[next_node] += r['mem']
            else: remaining.append(r)
        running[next_node] = remaining

        scheduled = True
        while scheduled and queue:
            scheduled = False
            for qtask in queue[:]:
                act_strat_q = strategy
                if is_ml:
                    row_dict = {'cpu_req': qtask['cpu'], 'mem_req': qtask['mem'], 'max_delay': qtask['max_delay'], 'queue_len': len(queue), 'total_cpu': total_cpu_capacity}
                    for i, fc in enumerate(free_cpu): row_dict[f'free_cpu_node{i}'] = fc
                    X_in = [[row_dict[col] for col in feature_cols]]
                    act_strat_q = 'BFRA' if clf.predict(X_in)[0] == 0 else 'FFRR'
                
                assigned = False
                if act_strat_q == 'BFRA':
                    best_idx, best_left = None, None
                    for i in range(num_nodes):
                        if free_cpu[i] >= qtask['cpu'] and free_mem[i] >= qtask['mem']:
                            slack = free_cpu[i] - qtask['cpu']
                            if best_idx is None or slack < best_left: best_idx, best_left = i, slack
                    if best_idx is not None: assign_task(best_idx, qtask, current_time); queue.remove(qtask); assigned = True
                else:
                    for offset in range(num_nodes):
                        i = (pointer + offset) % num_nodes
                        if free_cpu[i] >= qtask['cpu'] and free_mem[i] >= qtask['mem']:
                            assign_task(i, qtask, current_time); queue.remove(qtask); pointer = (i + 1) % num_nodes; assigned = True; break
                if assigned: scheduled = True

    return finish_times, execution_log, queue_history

def evaluate_strategy(tasks, finish_times):
    completion_times = [finish_times[t['id']] - t['arrival'] for t in tasks]
    return np.mean(completion_times)

# --- DASHBOARD UI ---
st.sidebar.markdown("### Simulation Parameters")
num_episodes = st.sidebar.slider("Episodes for Avg Testing", 5, 50, 20)
tasks_per_episode = st.sidebar.slider("Tasks per Episode", 10, 100, 30)
arrival_interval = st.sidebar.slider("Arrival Interval (Seconds)", 0.1, 5.0, 1.0)
max_cpu = st.sidebar.slider("Max CPU Required per Task", 10, 100, 50)

nodes = [{'cpu':100, 'mem':1000} for _ in range(3)]

st.write("### 1. Overall System Performance (Averages)")
st.markdown("""
To ensure reliability, we run multiple simulation episodes and calculate the **Average Completion Time** of all tasks. 
A lower time means the network is processing IoT data faster and preventing bottlenecks. The ML model predicts the most efficient strategy (BFRA or FFRR) for every single task based on real-time network states.
""")

if st.button("Run Simulation & Visualizations", type="primary"):
    with st.spinner("Simulating Edge Network Traffic..."):
        # Run Multiple Episodes
        bfra_cts, ffrr_cts, ml_cts = [], [], []
        for ep in range(num_episodes):
            tasks = generate_task_stream(tasks_per_episode, arrival_interval, (10, max_cpu))
            f_bfra, _, _ = simulate_with_logs(tasks, nodes, 'BFRA')
            f_ffrr, _, _ = simulate_with_logs(tasks, nodes, 'FFRR')
            f_ml, _, _ = simulate_with_logs(tasks, nodes, 'ML', is_ml=True, clf=clf, feature_cols=feature_cols)

            bfra_cts.append(evaluate_strategy(tasks, f_bfra))
            ffrr_cts.append(evaluate_strategy(tasks, f_ffrr))
            ml_cts.append(evaluate_strategy(tasks, f_ml))

        # Show KPIs
        col1, col2, col3 = st.columns(3)
        col1.metric("BFRA Avg Completion Time", f"{np.mean(bfra_cts):.2f}s")
        col2.metric("FFRR Avg Completion Time", f"{np.mean(ffrr_cts):.2f}s")
        col3.metric("ML (Dynamic) Completion Time", f"{np.mean(ml_cts):.2f}s")

        st.divider()

        # Run ONE Detailed Episode
        st.write("### 2. Deep Dive: Task Scheduling Visualization")
        st.markdown("""
        These Gantt charts show exactly *where* and *when* a stream of tasks was executed across our 3 Edge Nodes. 
        - **BFRA** tries to pack tasks tightly where they fit best.
        - **FFRR** spreads tasks out sequentially (Round-Robin).
        - **ML Strategy** dynamically switches between both to prevent idle time.
        """)
        
        detailed_tasks = generate_task_stream(tasks_per_episode, arrival_interval, (10, max_cpu))
        _, log_bfra, q_bfra = simulate_with_logs(detailed_tasks, nodes, 'BFRA')
        _, log_ffrr, q_ffrr = simulate_with_logs(detailed_tasks, nodes, 'FFRR')
        _, log_ml, q_ml = simulate_with_logs(detailed_tasks, nodes, 'ML', is_ml=True, clf=clf, feature_cols=feature_cols)
        
        def plot_gantt(log_data, title):
            df_g = pd.DataFrame(log_data)
            fig = px.bar(df_g, base="Start", x="Duration", y="Node", color="Task", orientation='h', title=title)
            fig.update_layout(showlegend=False, xaxis_title="Simulation Time (s)", yaxis_title="")
            return fig

        g_col1, g_col2, g_col3 = st.columns(3)
        with g_col1: st.plotly_chart(plot_gantt(log_bfra, "BFRA Strategy"), use_container_width=True)
        with g_col2: st.plotly_chart(plot_gantt(log_ffrr, "FFRR Strategy"), use_container_width=True)
        with g_col3: st.plotly_chart(plot_gantt(log_ml, "ML Dynamic Strategy"), use_container_width=True)

        st.divider()

        # Queue Length Line Chart
        st.write("### 3. Network Congestion Tracking")
        st.markdown("""
        This chart tracks the number of tasks stuck waiting in the queue over time. When edge nodes run out of CPU/Memory capacity, the queue length spikes, causing high latency. A smoother, lower line indicates better resource allocation.
        """)
        df_q_bfra = pd.DataFrame(q_bfra); df_q_bfra['Strategy'] = 'BFRA'
        df_q_ffrr = pd.DataFrame(q_ffrr); df_q_ffrr['Strategy'] = 'FFRR'
        df_q_ml = pd.DataFrame(q_ml); df_q_ml['Strategy'] = 'ML'
        
        df_q_all = pd.concat([df_q_bfra, df_q_ffrr, df_q_ml])
        fig_q = px.line(df_q_all, x="Time", y="Queue Length", color="Strategy", markers=True)
        st.plotly_chart(fig_q, use_container_width=True)

        st.divider()

        # Feature Importance Placeholder for XAI
        st.write("### 4. Explainable AI (XAI): Model Transparency")
        st.markdown("""
        To avoid the "Black Box" problem, we must understand *why* the Machine Learning model makes its choices. 
        The chart below shows the Random Forest's built-in feature importance—revealing exactly which real-time network conditions (like Queue Length or the CPU load on a specific node) carry the most weight when the AI decides between BFRA and FFRR[cite: 33, 34]. 
        *(Note: In the next phase, we will integrate SHAP for deeper task-by-task transparency).*
        """)
        importances = clf.feature_importances_
        df_imp = pd.DataFrame({'System Feature': feature_cols, 'Decision Importance': importances}).sort_values('Decision Importance', ascending=True)
        fig_imp = px.bar(df_imp, x='Decision Importance', y='System Feature', orientation='h')
        fig_imp.update_traces(marker_color='#1f77b4')
        st.plotly_chart(fig_imp, use_container_width=True)