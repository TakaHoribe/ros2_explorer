import dash
from dash import html, dcc
from dash.dependencies import Input, Output
import rclpy
from rclpy.node import Node
import re
import time
import subprocess
import threading
import logging
import argparse
import webbrowser

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import html_layout

# Set up argument parser
parser = argparse.ArgumentParser(description='ROS 2 Node and Topic Explorer')
parser.add_argument('--hz_all', action='store_true', help='Enable Hz update for node page')
parser.add_argument('--no-browser', action='store_true', help='Do not automatically open the web browser')
args = parser.parse_args()

# Configure logging
logging.basicConfig(level=logging.INFO, format='[%(asctime)s] %(message)s', datefmt='%Y-%m-%d %H:%M:%S')

# ===================================================================
# ======================= ros2 command thread =======================
# ===================================================================

class ThreadManager:
    def __init__(self):
        self.thread_controls = {'hz': {}, 'echo': {}, 'param': {}}

    def clear_threads(self, command_type, exclude_topic=None):
        """
        Clear separated threads for a specific command type.
        """
        topics_to_remove = []

        for topic, control in self.thread_controls[command_type].items():
            if topic == exclude_topic:
                continue
            control['stop'] = True
            if control['process']:
                control['process'].terminate()
            topics_to_remove.append(topic)

        # Remove all topics except the exclude_topic
        for topic in topics_to_remove:
            del self.thread_controls[command_type][topic]

    def start_threads(self, command_type, topic_name, func):
        """
        Start separated threads for a specific command type and target function.
        """
        self.thread_controls[command_type][topic_name] = {
            'stop': False, 'process': None, 'messages': [], 'display': True}
        thread = threading.Thread(target=func, args=(
            topic_name, self.thread_controls[command_type][topic_name]), daemon=True)
        thread.start()

thread_manager = ThreadManager()

def run_command(command, control_dict, max_line=1000):
    """
    Run a ROS 2 command in a separate thread and store its output in a message list.
    """
    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, shell=True, executable='/bin/bash', text=True)
    control_dict['process'] = process

    while True:
        output = process.stdout.readline()
        if output and control_dict['display']:
            msg = output.rstrip('\n')
            control_dict['messages'].append(msg)
            if len(control_dict['messages']) > max_line:
                control_dict['messages'].pop(0)
        else:
            time.sleep(0.1)

        if control_dict['stop']:
            logging.info(f"Stopping thread for command: {command}")
            break

def run_ros2_param(node_name, control_dict):
    run_command(f"ros2 param dump {node_name}", control_dict)

def run_ros2_echo(topic_name, control_dict, loop=False):
    option = "" if loop else " --once"
    run_command(f"ros2 topic echo {topic_name}{option}", control_dict)

def run_ros2_hz(topic_name, control_dict):
    run_command(f"ros2 topic hz {topic_name}", control_dict, max_line=100)

# ===================================================================
# ======================= for ros2 interface ========================
# ===================================================================
class ROS2InfoNode(Node):
    def __init__(self):
        super().__init__('ros2_info_node')

    def get_all_topics(self):
        return self.get_topic_names_and_types()

    def get_all_nodes(self):
        combined_names = []
        for name, namespace in self.get_node_names_and_namespaces():
            if name.startswith("_"):
                continue  # skip for _ros2cli, etc
            combined_name = f"{namespace}/{name}" if namespace != '/' else f"/{name}"
            combined_names.append(combined_name)
        return combined_names

    def get_topics_by_node(self, node_name):
        name, namespace = self.extract_name_and_namespace(node_name)
        pubs = self.get_publisher_names_and_types_by_node(name, namespace)
        subs = self.get_subscriber_names_and_types_by_node(name, namespace)
        return {'publishers': pubs, 'subscribers': subs}

    def get_nodes_by_topic(self, topic_name):
        pub_nodes = self.get_publishers_info_by_topic(topic_name)
        sub_nodes = self.get_subscriptions_info_by_topic(topic_name)
        publishers = [self.format_node_name(pub) for pub in pub_nodes if self.format_node_name(pub) is not None]
        subscribers = [self.format_node_name(sub) for sub in sub_nodes if self.format_node_name(sub) is not None]
        return {'publishers': publishers, 'subscribers': subscribers}

    @staticmethod
    def extract_name_and_namespace(full_name):
        parts = full_name.split('/')
        return parts[-1], '/'.join(parts[:-1])

    @staticmethod
    def format_node_name(node_info):
        if node_info.node_name.startswith("_"):  # skip for _ros2cli, etc
            return None
        return f"{node_info.node_namespace}/{node_info.node_name}" if node_info.node_namespace != '/' else f"/{node_info.node_name}"

# Initialize ROS2
rclpy.init()
ros2_node = ROS2InfoNode()

# Initialize Dash app with external stylesheet
external_stylesheets = ['https://fonts.googleapis.com/css2?family=Lato:wght@300;700&display=swap']
app = dash.Dash(__name__, external_stylesheets=external_stylesheets)
app.title = 'ROS 2 Explorer'

# wait for ros connection, otherwise ros information is not available
time.sleep(1)

app.layout = html_layout.generate_main_layout()

# ===================================================================
# ========= For main (ros2 node /ros2 topic) info division ==========
# ===================================================================
@app.callback(
    Output('main-info', 'children'),
    Input('url', 'pathname')
)
def update_main_info(pathname):

    """
    Update topic information based on the URL pathname.

    Args:
    pathname (str): The URL pathname used to extract the topic or node name.

    Returns:
    html.Div: A Div containing the ROS 2 topic or node information.
    """

    def extract_name(name_key):
        """Extract the name from the pathname using a specified key."""
        return pathname.split(f'{name_key}')[-1]

    if 'topic_name=' in pathname:
        topic_name = extract_name('topic_name=')

        # Stop the existing thread for the old topic and start a new one for the new topic
        thread_manager.clear_threads('hz')

        thread_manager.clear_threads('hz', exclude_topic=topic_name)
        if topic_name not in thread_manager.thread_controls['hz']:
            thread_manager.start_threads('hz', topic_name, run_ros2_hz)

        thread_manager.clear_threads('echo')
        if topic_name not in thread_manager.thread_controls['echo']:
            thread_manager.start_threads('echo', topic_name, run_ros2_echo)

        else:
            logging.info(
                f"update_topic_info: topic_name {topic_name} is in thread_control.")
 
        return generate_main_topic_info_div(topic_name)
    
    elif 'node_name=' in pathname:
        node_name = extract_name('node_name=')

        # start topic hz thread for all pubsub topics
        topics = ros2_node.get_topics_by_node(node_name)
        
        thread_manager.clear_threads('hz')
        if args.hz_all:
            for topic in topics['publishers'] + topics['subscribers']:
                topic_name = topic[0]
                if topic_name not in thread_manager.thread_controls['hz']:
                    thread_manager.start_threads('hz', topic_name, run_ros2_hz)
        
        thread_manager.clear_threads('param')
        if node_name not in thread_manager.thread_controls['param']:
            thread_manager.start_threads('param', node_name, run_ros2_param)

        return generate_main_node_info_div(node_name, topics)

    else:
        logging.info(f"UNKNOWN mode: URL = {pathname}")

    return html.H2("Select a topic or node to see the information.")



# ===================================================================
# ======================== Topic list update ========================
# ===================================================================
def get_topic_list(pattern=None):
    topics = ros2_node.get_all_topics()
    if pattern:
        regex_pattern = re.compile(pattern)
        topics = filter(lambda topic: regex_pattern.search(topic[0]), topics)
    return [{'topic_name_type': f"{topic} [{topic_type[0]}]", 'name': topic} for topic, topic_type in topics]

@app.callback(
    Output('topic-list', 'children'),
    Input('interval-update-node-topic-list', 'n_intervals'),
    Input('topic-filter', 'value')
)
def update_topic_list(n_intervals, topic_pattern):
    topic_options = get_topic_list(topic_pattern)
    topic_list = html.Ul([
        html.Li([
            html.A(topic_option['topic_name_type'],
                   href=f"/topic_name={topic_option['name']}")
        ]) for topic_option in topic_options])
    return topic_list

# ===================================================================
# ========================= Node list update ========================
# ===================================================================
def get_node_list(pattern=None):
    nodes = ros2_node.get_all_nodes()
    if pattern:
        regex_pattern = re.compile(pattern)
        nodes = filter(lambda node: regex_pattern.search(node), nodes)
    return sorted([f"{node}" for node in nodes])

@app.callback(
    Output('node-list', 'children'),
    Input('interval-update-node-topic-list', 'n_intervals'),
    Input('node-filter', 'value')
)
def update_node_list(n_intervals, node_pattern):
    node_names = get_node_list(node_pattern)
    node_list = html.Ul([
        html.Li(html.A(node, href=f"/node_name={node}")) for node in node_names
    ])
    return node_list


# ===================================================================
# ================== For ros2 topic info division ===================
# ===================================================================
def generate_main_topic_info_div(topic_name):
        nodes = ros2_node.get_nodes_by_topic(topic_name)

        def create_node_links(node_list):
            """Create HTML links for each node in the node list."""
            sorted_node_list = sorted(node_list)
            return [html.A(f"{node}", href=f"/node_name={node}") for node in sorted_node_list]

        publishers, subscribers = create_node_links(nodes['publishers']), create_node_links(nodes['subscribers'])

        return html_layout.generate_pubsub_layout(topic_name, publishers, subscribers)

def update_thread_output(thread_control, pathname):
    if 'topic_name=' in pathname:
        topic_name = pathname.split('topic_name=')[-1]
        if topic_name in thread_control:
            control_dict = thread_control[topic_name]
            if control_dict['messages']:
                return "\n".join(control_dict['messages'])
            else:
                return "No output received..."
        else:
            logging.info(
                f"topic_name = {topic_name} is NOT in thread control. thread_control = {thread_control}")

    if 'node_name=' in pathname:
        node_name = pathname.split('node_name=')[-1]
        if node_name in thread_control:
            control_dict = thread_control[node_name]
            if control_dict['messages']:
                return "\n".join(control_dict['messages'])
            else:
                return "No output received..."
        else:
            logging.info(
                f"node_name = {node_name} is NOT in thread control. thread_control = {thread_control}")

    return 'None'

# --- hz update in topic info page ---
@app.callback(
    Output('hz-output', 'children'),
    [Input('update-interval-topic-page-hz', 'n_intervals')],
    [dash.dependencies.State('url', 'pathname')]
)
def update_hz_output(n_intervals, pathname):
    return update_thread_output(thread_manager.thread_controls['hz'], pathname)


# --- hz update in topic info page ---
@app.callback(
    Output('echo-output', 'children'),
    [Input('update-interval-topic-page-echo', 'n_intervals')],
    [dash.dependencies.State('url', 'pathname')]
)
def update_hz_output(n_intervals, pathname):
    return update_thread_output(thread_manager.thread_controls['echo'], pathname)


# --- Callback for the "once" button in topic info page ---
@app.callback(
    Output('dummy-output', 'children'),  # Using a dummy output as no actual output is needed
    Input('echo-reset-button', 'n_clicks'),
    [dash.dependencies.State('url', 'pathname')]
)
def reset_echo(n_clicks, pathname):
    if 'topic_name=' in pathname and n_clicks > 0:  # Check if the button was clicked
        topic_name = pathname.split('topic_name=')[-1]

        # Clear the echo threads and start a new one
        thread_manager.clear_threads('echo')
        thread_manager.start_threads('echo', topic_name, run_ros2_echo)

    # Return is required for a callback, but in this case, it does nothing
    return None

# --- Callback for the "loop" button in topic info page ---
@app.callback(
    Output('loop-button', 'style'),
    Input('loop-button', 'n_clicks'),
    [dash.dependencies.State('url', 'pathname')]
)
def toggle_loop(n_clicks, pathname):
    # Toggle state: odd clicks mean looping, even clicks mean not looping. 0 means initial time.
    clicked = n_clicks % 2 == 1
    if 'topic_name=' in pathname:
        topic_name = pathname.split('topic_name=')[-1]

        if clicked:
            thread_manager.clear_threads('echo')
            thread_manager.start_threads('echo', topic_name, lambda tn, ctrl_dict: run_ros2_echo(tn, ctrl_dict, loop=True))
        elif n_clicks != 0:  # if this is not the initial time
            thread_manager.thread_controls['echo'][topic_name]['display'] = False

    # Change color based on whether it's an odd or even click
    return {'backgroundColor': 'darkgray'} if clicked else {}


# ===================================================================
# =================== For ros2 node info division ===================
# ===================================================================
def generate_main_node_info_div(node_name, topics):

    publishers = [html.A(f"{topic[0]} [{topic[1][0]}]", href=f"/topic_name={topic[0]}")
                    for topic in topics['publishers']]
    subscribers = [html.A(f"{topic[0]} [{topic[1][0]}]", href=f"/topic_name={topic[0]}")
                    for topic in topics['subscribers']]

    pre_style = {'word-wrap': 'break-word', 'white-space': 'pre-wrap', 'margin-left': '1rem', 'margin-right': '1rem'}

    children = [
        html.Div([
        html.H3(f"ros2 node info {node_name}", style={'margin-left': '1em'}),
        html.Div([
            html.H4("Publishing Topics:", style={'margin-left': '2em'}),
            html.Ul([html.Li(publisher) for publisher in publishers], id='publishers-list'),
            html.H4("Subscribing Topics:", style={'margin-left': '2em'}),
            html.Ul([html.Li(subscriber) for subscriber in subscribers], id='subscribers-list'),
        ], style={'overflowY': 'auto', 'maxHeight': '30em', 'margin': '1em'}),
        ], style={'width': '70%', 'float': 'left'}),
        html.Div([
            html.H3("ros2 param dump", style={'margin-left': '1em'}),
            html.Div([
                html.Pre(id='param-output', children='Loading...', style=pre_style),
            ], style={'overflowY': 'auto', 'maxHeight': '30em', 'margin': '1em'}),
        ], style={'width': '30%', 'float': 'left'}),
        dcc.Interval(id='update-interval-node-page-param', interval=200, n_intervals=0),
    ]

    # Conditionally add Interval component
    if args.hz_all:
        children.append(dcc.Interval(id='interval-update-node-page-pubsub-hz', interval=200, n_intervals=0))

    return html.Div(children)

# --- param update in node info page ---
@app.callback(
    Output('param-output', 'children'),
    [Input('update-interval-node-page-param', 'n_intervals')],
    [dash.dependencies.State('url', 'pathname')]
)
def update_hz_output(n_intervals, pathname):
    return update_thread_output(thread_manager.thread_controls['param'], pathname)


def extract_rate_from_hz_message(topic_name):
    hz_control = thread_manager.thread_controls.get('hz', {})
    topic_control = hz_control.get(topic_name, {})
    hz_messages = topic_control.get('messages', [])

    if hz_messages:
        for message in reversed(hz_messages):  # Iterate in reverse to start from the most recent message
            match = re.search(r"average rate: ([0-9.]+)", message)
            if match:
                return float(match.group(1))
    return '-'

# --- Update node pubsub hz ---
@app.callback(
    [Output('publishers-list', 'children'),
     Output('subscribers-list', 'children')],
    Input('interval-update-node-page-pubsub-hz', 'n_intervals'),
    [dash.dependencies.State('url', 'pathname')]
)
def update_node_pubsub_hz(n_intervals, pathname):
    def extract_name(name_key):
        """Extract the name from the pathname using a specified key."""
        return pathname.split(f'{name_key}')[-1]
    

    logging.info(f"thread_controls len = {len(thread_manager.thread_controls['echo'])}, {len(thread_manager.thread_controls['hz'])}")

    if 'node_name=' in pathname:
        node_name = extract_name('node_name=')
        topics = ros2_node.get_topics_by_node(node_name)

        updated_publishers = [
            html.Li(html.A(f"{topic[0]} [{topic[1][0]}] ({extract_rate_from_hz_message(topic[0])} Hz)",
                           href=f"/topic_name={topic[0]}"))
            for topic in topics['publishers']
        ]
        updated_subscribers = [
            html.Li(html.A(f"{topic[0]} [{topic[1][0]}] ({extract_rate_from_hz_message(topic[0])} Hz)",
                           href=f"/topic_name={topic[0]}"))
            for topic in topics['subscribers']
        ]

        return updated_publishers, updated_subscribers

    return [], []  # Return empty lists if pathname does not include 'node_name='

def main():
    # Open the web browser to the Dash app URL
    if not args.no_browser:
        webbrowser.open('http://127.0.0.1:8050/')

    # Run application
    app.run_server(debug=False)

if __name__ == '__main__':
    main()
