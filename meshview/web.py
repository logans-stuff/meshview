"""Main web server routes and page rendering for Meshview."""

import asyncio
import datetime
import logging
import os
import pathlib
import re
import ssl
from dataclasses import dataclass

import pydot
from aiohttp import web
from google.protobuf import text_format
from google.protobuf.message import Message
from jinja2 import Environment, PackageLoader, Undefined, select_autoescape
from markupsafe import Markup

from meshtastic.protobuf.portnums_pb2 import PortNum
from meshview import config, database, decode_payload, migrations, models, store
from meshview.__version__ import (
    __version_string__,
)
from meshview.web_api import api

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(filename)s:%(lineno)d [pid:%(process)d] %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

logger = logging.getLogger(__name__)
SEQ_REGEX = re.compile(r"seq \d+")
SOFTWARE_RELEASE = __version_string__  # Keep for backward compatibility
CONFIG = config.CONFIG

env = Environment(loader=PackageLoader("meshview"), autoescape=select_autoescape())

# Start Database
database.init_database(CONFIG["database"]["connection_string"])

BASE_DIR = os.path.dirname(__file__)
LANG_DIR = os.path.join(BASE_DIR, "lang")

with open(os.path.join(os.path.dirname(__file__), '1x1.png'), 'rb') as png:
    empty_png = png.read()


@dataclass
class Packet:
    """UI-friendly packet wrapper for templates and API payloads."""

    id: int
    from_node_id: int
    from_node: models.Node
    to_node_id: int
    to_node: models.Node
    portnum: int
    data: str
    raw_mesh_packet: object
    raw_payload: object
    payload: str
    pretty_payload: Markup
    import_time_us: int

    @classmethod
    def from_model(cls, packet):
        """Convert a Packet ORM model into a presentation-friendly Packet."""
        mesh_packet, payload = decode_payload.decode(packet)
        pretty_payload = None

        if mesh_packet:
            mesh_packet.decoded.payload = b""
            text_mesh_packet = text_format.MessageToString(mesh_packet)
        else:
            text_mesh_packet = "Did node decode"

        if payload is None:
            text_payload = "Did not decode"
        elif isinstance(payload, Message):
            text_payload = text_format.MessageToString(payload)
        elif packet.portnum == PortNum.TEXT_MESSAGE_APP and packet.to_node_id != 0xFFFFFFFF:
            text_payload = "<redacted>"
        elif isinstance(payload, bytes):
            text_payload = payload.decode("utf-8", errors="replace")  # decode bytes safely
        else:
            text_payload = str(payload)

        if payload:
            if (
                packet.portnum == PortNum.POSITION_APP
                and getattr(payload, "latitude_i", None)
                and getattr(payload, "longitude_i", None)
            ):
                pretty_payload = Markup(
                    f'<a href="https://www.google.com/maps/search/?api=1&query={payload.latitude_i * 1e-7},{payload.longitude_i * 1e-7}" target="_blank">map</a>'
                )

        return cls(
            id=packet.id,
            from_node=packet.from_node,
            from_node_id=packet.from_node_id,
            to_node=packet.to_node,
            to_node_id=packet.to_node_id,
            portnum=packet.portnum,
            data=text_mesh_packet,
            payload=text_payload,  # now always a string
            pretty_payload=pretty_payload,
            import_time_us=packet.import_time_us,  # <-- include microseconds
            raw_mesh_packet=mesh_packet,
            raw_payload=payload,
        )


async def build_trace(node_id):
    """Build a recent GPS trace list for a node using position packets."""
    trace = []
    for raw_p in await store.get_packets_from(
        node_id, PortNum.POSITION_APP, since=datetime.timedelta(hours=24)
    ):
        p = Packet.from_model(raw_p)
        if not p.raw_payload or not p.raw_payload.latitude_i or not p.raw_payload.longitude_i:
            continue
        trace.append((p.raw_payload.latitude_i * 1e-7, p.raw_payload.longitude_i * 1e-7))

    if not trace:
        for raw_p in await store.get_packets_from(node_id, PortNum.POSITION_APP):
            p = Packet.from_model(raw_p)
            if not p.raw_payload or not p.raw_payload.latitude_i or not p.raw_payload.longitude_i:
                continue
            trace.append((p.raw_payload.latitude_i * 1e-7, p.raw_payload.longitude_i * 1e-7))
            break

    return trace


async def build_neighbors(node_id):
    """Return neighbor node metadata for the given node ID."""
    packets = await store.get_packets_from(node_id, PortNum.NEIGHBORINFO_APP, limit=1)
    packet = packets.first()

    if not packet:
        return []

    _, payload = decode_payload.decode(packet)
    neighbors = {}

    # Gather node information asynchronously
    tasks = {n.node_id: store.get_node(n.node_id) for n in payload.neighbors}
    results = await asyncio.gather(*tasks.values(), return_exceptions=True)

    for neighbor, node in zip(payload.neighbors, results, strict=False):
        if isinstance(node, Exception):
            continue
        if node and node.last_lat and node.last_long:
            neighbors[neighbor.node_id] = {
                'node_id': neighbor.node_id,
                'snr': neighbor.snr,  # Fix dictionary keying issue
                'short_name': node.short_name,
                'long_name': node.long_name,
                'location': (node.last_lat * 1e-7, node.last_long * 1e-7),
            }

    return list(neighbors.values())  # Return a list of dictionaries


def node_id_to_hex(node_id):
    """Format a node_id in Meshtastic hex notation."""
    if node_id is None or isinstance(node_id, Undefined):
        return "Invalid node_id"  # i... have no clue
    if node_id == 4294967295:
        return "^all"
    else:
        return f"!{hex(node_id)[2:].zfill(8)}"


def format_timestamp(timestamp):
    """Normalize timestamps to ISO 8601 strings."""
    if isinstance(timestamp, int):
        timestamp = datetime.datetime.fromtimestamp(timestamp, datetime.UTC)
    return timestamp.isoformat(timespec="milliseconds")


env.filters["node_id_to_hex"] = node_id_to_hex
env.filters["format_timestamp"] = format_timestamp

# Initialize API module with dependencies
api.init_api_module(Packet, SEQ_REGEX, LANG_DIR)

# Create main routes table
routes = web.RouteTableDef()


@routes.get("/")
async def index(request):
    """Redirect root URL to configured starting page."""
    """
    Redirect root URL '/' to the page specified in CONFIG['site']['starting'].
    Defaults to '/map' if not set.
    """
    # Get the starting page from config
    starting_url = CONFIG["site"].get("starting", "/map")  # default to /map if not set
    raise web.HTTPFound(location=starting_url)


# redirect for backwards compatibility
@routes.get("/packet_list/{packet_id}")
async def redirect_packet_list(request):
    packet_id = request.match_info["packet_id"]
    raise web.HTTPFound(location=f"/node/{packet_id}")

# Generic static HTML route
@routes.get("/{page}")
async def serve_page(request):
    """Serve static HTML pages from meshview/static."""
    page = request.match_info["page"]

    # default to index.html if no extension
    if not page.endswith(".html"):
        page = f"{page}.html"

    html_file = pathlib.Path(__file__).parent / "static" / page
    if not html_file.exists():
        raise web.HTTPNotFound(text=f"Page '{page}' not found")

    content = html_file.read_text(encoding="utf-8")
    return web.Response(text=content, content_type="text/html")


@routes.get("/net")
async def net(request):
    return web.Response(
        text=env.get_template("net.html").render(),
        content_type="text/html",
    )


@routes.get("/map")
async def map(request):
    template = env.get_template("map.html")
    return web.Response(text=template.render(), content_type="text/html")


@routes.get("/nodelist")
async def nodelist(request):
    template = env.get_template("nodelist.html")
    return web.Response(
        text=template.render(),
        content_type="text/html",
    )


@routes.get("/firehose")
async def firehose(request):
    return web.Response(
        text=env.get_template("firehose.html").render(),
        content_type="text/html",
    )


@routes.get("/chat")
async def chat(request):
    template = env.get_template("chat.html")
    return web.Response(
        text=template.render(),
        content_type="text/html",
    )


@routes.get("/packet/{packet_id}")
async def new_packet(request):
    template = env.get_template("packet.html")
    return web.Response(
        text=template.render(),
        content_type="text/html",
    )


@routes.get("/node/{from_node_id}")
async def firehose_node(request):
    template = env.get_template("node.html")
    return web.Response(
        text=template.render(),
        content_type="text/html",
    )


@routes.get("/nodegraph")
async def nodegraph(request):
    template = env.get_template("nodegraph.html")
    return web.Response(
        text=template.render(),
        content_type="text/html",
    )


@routes.get("/top")
async def top(request):
    template = env.get_template("top.html")
    return web.Response(
        text=template.render(),
        content_type="text/html",
    )


@routes.get("/stats")
async def stats(request):
    template = env.get_template("stats.html")
    return web.Response(
        text=template.render(),
        content_type="text/html",
    )


# Keep !!
@routes.get("/graph/traceroute/{packet_id}")
async def graph_traceroute(request):
    packet_id = int(request.match_info['packet_id'])
    traceroutes = list(await store.get_traceroute(packet_id))

    packet = await store.get_packet(packet_id)
    if not packet:
        return web.Response(
            status=404,
        )

    # Find related packets (request/response pairs) - strict matching
    related_packets = []
    if packet.from_node_id and packet.to_node_id and packet.import_time_us:
        # Only look within ±5 minutes of current packet
        time_window_us = 5 * 60 * 1_000_000  # 5 minutes in microseconds
        time_start = packet.import_time_us - time_window_us
        time_end = packet.import_time_us + time_window_us

        # Find packets with swapped from/to within time window
        related = await store.get_packets(
            from_node_id=packet.to_node_id,
            to_node_id=packet.from_node_id,
            portnum=70,  # TRACEROUTE_APP
            after=time_start,
            limit=20
        )

        for rel_pkt in related:
            # Skip self and packets outside time window
            if rel_pkt.id == packet_id or rel_pkt.import_time_us > time_end:
                continue

            # Only include if this packet has traceroute data
            rel_traceroutes = list(await store.get_traceroute(rel_pkt.id))
            if rel_traceroutes:
                direction = "Response Packet" if rel_pkt.from_node_id == packet.to_node_id else "Request Packet"
                related_packets.append({
                    'id': rel_pkt.id,
                    'direction': direction
                })
                # Only show the first related packet in each direction
                if len(related_packets) >= 2:
                    break

    node_ids = set()
    for tr in traceroutes:
        route = decode_payload.decode_payload(PortNum.TRACEROUTE_APP, tr.route)
        node_ids.add(tr.gateway_node_id)
        for node_id in route.route:
            node_ids.add(node_id)
        # Also collect nodes from return path
        if hasattr(route, 'route_back'):
            for node_id in route.route_back:
                node_ids.add(node_id)
        # Handle route_return field if present
        if tr.route_return:
            route_return = decode_payload.decode_payload(PortNum.TRACEROUTE_APP, tr.route_return)
            if route_return and hasattr(route_return, 'route'):
                for node_id in route_return.route:
                    node_ids.add(node_id)
    node_ids.add(packet.from_node_id)
    node_ids.add(packet.to_node_id)

    nodes = {}
    async with asyncio.TaskGroup() as tg:
        for node_id in node_ids:
            nodes[node_id] = tg.create_task(store.get_node(node_id))

    # FIRST: Determine the initiator before building the graph
    initiator_id = None
    target_id = None
    for tr in traceroutes:
        tr_packet = await store.get_packet(tr.packet_id)
        if tr_packet:
            if not tr.done:
                initiator_id = tr_packet.from_node_id
                target_id = tr_packet.to_node_id
                break
            elif tr.done and not initiator_id:
                initiator_id = tr_packet.to_node_id
                target_id = tr_packet.from_node_id
    if not initiator_id:
        initiator_id = packet.from_node_id
        target_id = packet.to_node_id

    graph = pydot.Dot('traceroute', graph_type="digraph")

    # Bright, readable color palette (avoiding black and dark colors)
    COLOR_PALETTE = [
        '#FF6B6B',  # Red
        '#4ECDC4',  # Teal
        '#45B7D1',  # Blue
        '#FFA07A',  # Light Salmon
        '#98D8C8',  # Mint
        '#F7DC6F',  # Yellow
        '#BB8FCE',  # Purple
        '#85C1E2',  # Sky Blue
        '#F8B739',  # Orange
        '#52C41A',  # Green
        '#FF85C0',  # Pink
        '#95E1D3',  # Aqua
    ]

    paths = set()
    node_color = {}
    mqtt_nodes = set()
    saw_reply = set()
    dest = None
    node_seen_time = {}

    # Track which packet each traceroute belongs to
    for tr in traceroutes:
        if tr.done:
            saw_reply.add(tr.gateway_node_id)
        if tr.done and dest:
            continue
        route = decode_payload.decode_payload(PortNum.TRACEROUTE_APP, tr.route)

        # Determine which packet this traceroute is for
        # If it's from a related packet, the path direction is different
        tr_packet = await store.get_packet(tr.packet_id)
        if not tr_packet:
            continue

        # Use the traceroute's packet's from/to nodes, not the current packet's
        path_start = tr_packet.from_node_id
        path_end = tr_packet.to_node_id

        # Process forward path (from traceroute's perspective)
        path = [path_start]
        path.extend(route.route)
        if tr.done:
            dest = path_end
            path.append(path_end)
        elif path[-1] != tr.gateway_node_id:
            # It seems some nodes add them self to the list before uplinking
            path.append(tr.gateway_node_id)

        if not tr.done and tr.gateway_node_id not in node_seen_time and tr.import_time_us:
            node_seen_time[path[-1]] = tr.import_time_us

        mqtt_nodes.add(tr.gateway_node_id)
        node_color[path[-1]] = COLOR_PALETTE[hash(tuple(path)) % len(COLOR_PALETTE)]
        paths.add(tuple(path))

        # Process return path (route_back) - direction is reversed
        if hasattr(route, 'route_back') and route.route_back:
            return_path = [path_end]  # Start from destination
            return_path.extend(route.route_back)
            return_path.append(path_start)  # End at source
            node_color[return_path[-1]] = COLOR_PALETTE[hash(tuple(return_path)) % len(COLOR_PALETTE)]
            paths.add(tuple(return_path))

        # Process route_return field if present
        if tr.route_return:
            route_return = decode_payload.decode_payload(PortNum.TRACEROUTE_APP, tr.route_return)
            if route_return and hasattr(route_return, 'route'):
                return_path_alt = [path_end]  # Start from destination
                return_path_alt.extend(route_return.route)
                return_path_alt.append(path_start)  # End at source
                node_color[return_path_alt[-1]] = COLOR_PALETTE[hash(tuple(return_path_alt)) % len(COLOR_PALETTE)]
                paths.add(tuple(return_path_alt))

    used_nodes = set()
    for path in paths:
        used_nodes.update(path)

    import_times = [tr.import_time_us for tr in traceroutes if tr.import_time_us]
    if import_times:
        first_time = min(import_times)
    else:
        first_time = 0

    for node_id in used_nodes:
        node = await nodes[node_id]
        if not node:
            node_name = node_id_to_hex(node_id)
        else:
            node_name = (
                f'[{node.short_name}] {node.long_name}\n{node_id_to_hex(node_id)}\n{node.role}'
            )
        if node_id in node_seen_time:
            ms = (node_seen_time[node_id] - first_time) / 1000
            node_name += f'\n {ms:.2f}ms'
        # Style priority: target > initiator > gateways > others
        style = 'dashed'
        penwidth = 1  # Default border width
        if node_id == target_id:
            style = 'filled'
        elif node_id == initiator_id:
            style = 'solid'
        elif node_id in mqtt_nodes:
            style = 'solid'

        # Always apply thick border to initiator, regardless of other styles
        if node_id == initiator_id:
            penwidth = 3

        if node_id in saw_reply:
            style += ', diagonals'

        graph.add_node(
            pydot.Node(
                str(node_id),
                label=node_name,
                shape='box',
                color=node_color.get(node_id, '#45B7D1'),  # Default to bright blue instead of black
                style=style,
                penwidth=penwidth,
                href=f"/node/{node_id}",
            )
        )

    for path in paths:
        color = COLOR_PALETTE[hash(tuple(path)) % len(COLOR_PALETTE)]
        for src, dest_node in zip(path, path[1:], strict=False):
            graph.add_edge(pydot.Edge(src, dest_node, color=color))

    # Build route analysis data
    # IMPORTANT: Identify the traceroute INITIATOR (who started the exchange)
    # The traceroute exchange has a request and response - we need to show
    # paths relative to whoever initiated it, not relative to current packet

    initiator_id = None
    target_id = None

    # Find the initiator by looking at traceroutes
    for tr in traceroutes:
        tr_packet = await store.get_packet(tr.packet_id)
        if tr_packet:
            if not tr.done:
                # This is a request/incomplete hop - this packet's sender is the initiator
                initiator_id = tr_packet.from_node_id
                target_id = tr_packet.to_node_id
                break
            elif tr.done and not initiator_id:
                # This is a completed response - sender is target, receiver is initiator
                initiator_id = tr_packet.to_node_id
                target_id = tr_packet.from_node_id

    # Fallback to current packet if we couldn't determine initiator
    if not initiator_id:
        initiator_id = packet.from_node_id
        target_id = packet.to_node_id

    forward_path_nodes = []
    return_path_nodes = []
    forward_complete = False
    return_complete = False

    # CRITICAL: Check if ANY traceroute has done=True
    # The done flag means the full round-trip completed (both forward and return)
    has_done = any(tr.done for tr in traceroutes)
    if has_done:
        # If done=True exists, both directions completed successfully
        forward_complete = True
        return_complete = True

    # Analyze paths relative to the INITIATOR (not current packet)
    for path in paths:
        # Forward direction: initiator → target
        if path[0] == initiator_id:
            if path[-1] == target_id:
                # Path reaches target - definitely complete
                forward_complete = True
                if not forward_path_nodes or len(path) > len(forward_path_nodes):
                    forward_path_nodes = list(path)
            elif not forward_path_nodes:
                # Incomplete forward path, use if we don't have one yet
                forward_path_nodes = list(path)

        # Return direction: target → initiator
        if path[0] == target_id:
            if path[-1] == initiator_id:
                # Path reaches initiator - definitely complete
                return_complete = True
                if not return_path_nodes or len(path) > len(return_path_nodes):
                    return_path_nodes = list(path)
            elif not return_path_nodes:
                # Incomplete return path, use if we don't have one yet
                return_path_nodes = list(path)

    # Convert node IDs to names for display
    async def get_node_name(node_id):
        node = await nodes.get(node_id)
        if node:
            return f"{node.short_name or node.long_name or node_id_to_hex(node_id)}"
        return node_id_to_hex(node_id)

    forward_path_display = []
    for node_id in forward_path_nodes:
        forward_path_display.append(await get_node_name(node_id))

    return_path_display = []
    for node_id in return_path_nodes:
        return_path_display.append(await get_node_name(node_id))

    # Determine routing status
    # IMPORTANT: Only mark as ASYMMETRIC when BOTH directions completed
    routing_status = None
    if forward_complete and return_complete:
        # Both directions completed - check if paths are the same
        forward_reversed = list(reversed(forward_path_nodes))
        if forward_reversed == return_path_nodes:
            routing_status = {'class': 'symmetric', 'text': 'SYMMETRIC ROUTING'}
        else:
            routing_status = {'class': 'asymmetric', 'text': 'ASYMMETRIC ROUTING'}
    elif forward_complete and not return_complete:
        routing_status = {'class': 'incomplete', 'text': 'NO RETURN PATH'}
    elif return_complete and not forward_complete:
        routing_status = {'class': 'incomplete', 'text': 'NO FORWARD PATH'}
    elif forward_path_nodes or return_path_nodes:
        routing_status = {'class': 'incomplete', 'text': 'INCOMPLETE'}
    else:
        routing_status = {'class': 'incomplete', 'text': 'NO PATH DATA'}

    # Get node names for packet info - use initiator/target, not current packet
    from_node = await nodes.get(initiator_id)
    to_node = await nodes.get(target_id)
    from_node_name = from_node.long_name if from_node else node_id_to_hex(initiator_id)
    to_node_name = to_node.long_name if to_node else node_id_to_hex(target_id)

    # Render template with analysis
    template = env.get_template('traceroute.html')
    html = template.render(
        packet_id=packet_id,
        from_node_name=from_node_name,
        to_node_name=to_node_name,
        graph_svg=graph.create_svg().decode('utf-8'),
        forward_path=forward_path_display if forward_path_display else None,
        return_path=return_path_display if return_path_display else None,
        forward_complete=forward_complete,
        return_complete=return_complete,
        routing_status=routing_status,
        related_packets=related_packets if related_packets else None
    )

    return web.Response(
        body=html,
        content_type="text/html",
    )


@routes.get("/graph/traceroute/{packet_id}/svg")
async def graph_traceroute_svg(request):
    """Return just the SVG graph for full-screen viewing"""
    packet_id = int(request.match_info['packet_id'])
    traceroutes = list(await store.get_traceroute(packet_id))

    packet = await store.get_packet(packet_id)
    if not packet:
        return web.Response(
            status=404,
        )

    # Build graph (reuse same logic but only return SVG)
    node_ids = set()
    for tr in traceroutes:
        route = decode_payload.decode_payload(PortNum.TRACEROUTE_APP, tr.route)
        node_ids.add(tr.gateway_node_id)
        for node_id in route.route:
            node_ids.add(node_id)
        if hasattr(route, 'route_back'):
            for node_id in route.route_back:
                node_ids.add(node_id)
        if tr.route_return:
            route_return = decode_payload.decode_payload(PortNum.TRACEROUTE_APP, tr.route_return)
            if route_return and hasattr(route_return, 'route'):
                for node_id in route_return.route:
                    node_ids.add(node_id)
    node_ids.add(packet.from_node_id)
    node_ids.add(packet.to_node_id)

    nodes = {}
    async with asyncio.TaskGroup() as tg:
        for node_id in node_ids:
            nodes[node_id] = tg.create_task(store.get_node(node_id))

    # FIRST: Determine the initiator before building the graph
    initiator_id = None
    target_id = None
    for tr in traceroutes:
        tr_packet = await store.get_packet(tr.packet_id)
        if tr_packet:
            if not tr.done:
                initiator_id = tr_packet.from_node_id
                target_id = tr_packet.to_node_id
                break
            elif tr.done and not initiator_id:
                initiator_id = tr_packet.to_node_id
                target_id = tr_packet.from_node_id
    if not initiator_id:
        initiator_id = packet.from_node_id
        target_id = packet.to_node_id

    graph = pydot.Dot('traceroute', graph_type="digraph")

    # Bright, readable color palette (avoiding black and dark colors)
    COLOR_PALETTE = [
        '#FF6B6B',  # Red
        '#4ECDC4',  # Teal
        '#45B7D1',  # Blue
        '#FFA07A',  # Light Salmon
        '#98D8C8',  # Mint
        '#F7DC6F',  # Yellow
        '#BB8FCE',  # Purple
        '#85C1E2',  # Sky Blue
        '#F8B739',  # Orange
        '#52C41A',  # Green
        '#FF85C0',  # Pink
        '#95E1D3',  # Aqua
    ]

    paths = set()
    node_color = {}
    mqtt_nodes = set()
    saw_reply = set()
    dest = None
    node_seen_time = {}

    for tr in traceroutes:
        if tr.done:
            saw_reply.add(tr.gateway_node_id)
        if tr.done and dest:
            continue
        route = decode_payload.decode_payload(PortNum.TRACEROUTE_APP, tr.route)

        tr_packet = await store.get_packet(tr.packet_id)
        if not tr_packet:
            continue

        path_start = tr_packet.from_node_id
        path_end = tr_packet.to_node_id

        path = [path_start]
        path.extend(route.route)
        if tr.done:
            dest = path_end
            path.append(path_end)
        elif path[-1] != tr.gateway_node_id:
            path.append(tr.gateway_node_id)

        if not tr.done and tr.gateway_node_id not in node_seen_time and tr.import_time_us:
            node_seen_time[path[-1]] = tr.import_time_us

        mqtt_nodes.add(tr.gateway_node_id)
        node_color[path[-1]] = COLOR_PALETTE[hash(tuple(path)) % len(COLOR_PALETTE)]
        paths.add(tuple(path))

        if hasattr(route, 'route_back') and route.route_back:
            return_path = [path_end]
            return_path.extend(route.route_back)
            return_path.append(path_start)
            node_color[return_path[-1]] = '#' + hex(hash(tuple(return_path)))[3:9]
            paths.add(tuple(return_path))

        if tr.route_return:
            route_return = decode_payload.decode_payload(PortNum.TRACEROUTE_APP, tr.route_return)
            if route_return and hasattr(route_return, 'route'):
                return_path_alt = [path_end]
                return_path_alt.extend(route_return.route)
                return_path_alt.append(path_start)
                node_color[return_path_alt[-1]] = '#' + hex(hash(tuple(return_path_alt)))[3:9]
                paths.add(tuple(return_path_alt))

    used_nodes = set()
    for path in paths:
        used_nodes.update(path)

    import_times = [tr.import_time_us for tr in traceroutes if tr.import_time_us]
    if import_times:
        first_time = min(import_times)
    else:
        first_time = 0

    for node_id in used_nodes:
        node = await nodes[node_id]
        if not node:
            node_name = node_id_to_hex(node_id)
        else:
            node_name = (
                f'[{node.short_name}] {node.long_name}\n{node_id_to_hex(node_id)}\n{node.role}'
            )
        if node_id in node_seen_time:
            ms = (node_seen_time[node_id] - first_time) / 1000
            node_name += f'\n {ms:.2f}ms'
        # Style priority: target > initiator > gateways > others
        style = 'dashed'
        penwidth = 1  # Default border width
        if node_id == target_id:
            style = 'filled'
        elif node_id == initiator_id:
            style = 'solid'
        elif node_id in mqtt_nodes:
            style = 'solid'

        # Always apply thick border to initiator, regardless of other styles
        if node_id == initiator_id:
            penwidth = 3

        if node_id in saw_reply:
            style += ', diagonals'

        graph.add_node(
            pydot.Node(
                str(node_id),
                label=node_name,
                shape='box',
                color=node_color.get(node_id, '#45B7D1'),  # Default to bright blue instead of black
                style=style,
                penwidth=penwidth,
                href=f"/node/{node_id}",
            )
        )

    for path in paths:
        color = COLOR_PALETTE[hash(tuple(path)) % len(COLOR_PALETTE)]
        for src, dest_node in zip(path, path[1:], strict=False):
            graph.add_edge(pydot.Edge(src, dest_node, color=color))

    # Return just the SVG
    return web.Response(
        body=graph.create_svg(),
        content_type="image/svg+xml",
    )


async def run_server():
    """Start the aiohttp web server after migrations are complete."""
    # Wait for database migrations to complete before starting web server
    logger.info("Checking database schema status...")
    database_url = CONFIG["database"]["connection_string"]

    # Wait for migrations to complete (writer app responsibility)
    migration_ready = await migrations.wait_for_migrations(
        database.engine, database_url, max_retries=30, retry_delay=2
    )

    if not migration_ready:
        logger.error("Database schema is not up to date. Cannot start web server.")
        raise RuntimeError("Database schema version mismatch - migrations not complete")

    logger.info("Database schema verified - starting web server")

    app = web.Application()
    app.router.add_static("/static/", pathlib.Path(__file__).parent / "static")
    app.add_routes(api.routes)  # Add API routes
    app.add_routes(routes)  # Add main web routes

    # Check if access logging should be disabled
    enable_access_log = CONFIG.get("logging", {}).get("access_log", "False").lower() == "true"
    access_log_handler = None if not enable_access_log else logging.getLogger("aiohttp.access")

    runner = web.AppRunner(app, access_log=access_log_handler)
    await runner.setup()
    if CONFIG["server"]["tls_cert"]:
        ssl_context = ssl.create_default_context(ssl.Purpose.CLIENT_AUTH)
        ssl_context.load_cert_chain(CONFIG["server"]["tls_cert"])
        logger.info(f"TLS enabled with certificate: {CONFIG['server']['tls_cert']}")
    else:
        ssl_context = None
        logger.info("TLS disabled")
    if host := CONFIG["server"]["bind"]:
        port = CONFIG["server"]["port"]
        protocol = "https" if ssl_context else "http"
        site = web.TCPSite(runner, host, port, ssl_context=ssl_context)
        await site.start()
        # Display localhost instead of wildcard addresses for usability
        display_host = "localhost" if host in ("0.0.0.0", "*", "::") else host
        logger.info(f"Web server started at {protocol}://{display_host}:{port}")
    while True:
        await asyncio.sleep(3600)  # sleep forever
