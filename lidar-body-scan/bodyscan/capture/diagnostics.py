"""Network diagnosis of the Ouster sensor (bodyscan check-sensor)."""

from __future__ import annotations

import socket
import subprocess
import sys
import time

from bodyscan.log import info


def _sdk():
    try:
        from ouster.sdk import core, sensor
    except ImportError as error:                     # pragma: no cover - depends on the installation
        raise SystemExit("the Ouster SDK is needed (ouster-sdk)") from error
    return core, sensor


def show_configuration(host: str):
    """Step 1: the sensor configuration over HTTP (no UDP involved)."""
    core, sensor = _sdk()
    info(f"[1] sensor configuration ({host})")
    try:
        config = sensor.get_config(host)
    except Exception as error:
        info(f"    cannot read the configuration: {error}")
        info("    The sensor is not reachable over HTTP: check cable, switch, IP address.")
        return None
    info(f"    udp_dest        : {config.udp_dest}")
    info(f"    udp_port_lidar  : {config.udp_port_lidar}")
    info(f"    udp_port_imu    : {config.udp_port_imu}")
    info(f"    operating_mode  : {config.operating_mode}")
    info(f"    lidar_mode      : {config.lidar_mode}")
    info(f"    udp_profile     : {config.udp_profile_lidar}")
    if config.operating_mode == core.OperatingMode.STANDBY:
        info("    -> the sensor is in STANDBY and sends nothing.")
    if config.udp_dest is None or ":" in str(config.udp_dest):
        info("    -> udp_dest is missing or an IPv6 address: set it to this PC (--set-ports ... --dest IP).")
    return config


def port_owner(port: int) -> str:
    """Windows only: which process owns a UDP port."""
    if not sys.platform.startswith("win"):
        return "unknown (not Windows)"
    command = ("Get-NetUDPEndpoint -LocalPort {p} -ErrorAction SilentlyContinue | "
               "ForEach-Object {{ $q = Get-Process -Id $_.OwningProcess -ErrorAction SilentlyContinue; "
               "'{{0}} (pid {{1}}) {{2}}' -f $q.ProcessName, $_.OwningProcess, $q.Path }}").format(p=port)
    try:
        result = subprocess.run(["powershell", "-NoProfile", "-Command", command], capture_output=True, text=True,
                                timeout=20)
        return result.stdout.strip() or "not listed (may need an administrator shell)"
    except Exception as error:
        return f"lookup failed: {error}"


def listen(port: int, seconds: float):
    """Bind a plain UDP socket and count what arrives: (bound, packets, sources)."""
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        sock.bind(("0.0.0.0", port))
    except OSError as error:
        info(f"    port {port}: CANNOT BIND ({error})")
        info(f"      owner: {port_owner(port)}")
        sock.close()
        return False, 0, set()
    sock.settimeout(0.5)
    packets, sources = 0, set()
    end = time.time() + seconds
    while time.time() < end:
        try:
            _, address = sock.recvfrom(65535)
            packets += 1
            sources.add(address[0])
        except socket.timeout:
            pass
    sock.close()
    info(f"    port {port}: bound, {packets} packets in {seconds:g} s"
         + (f" from {', '.join(sorted(sources))}" if sources else ""))
    return True, packets, sources


def test_sockets(config, seconds: float) -> None:
    """Step 2: plain UDP sockets on the sensor's ports, without the SDK."""
    info(f"[2] plain UDP socket test ({seconds:g} s per port, SDK not involved)")
    ports = [config.udp_port_lidar, config.udp_port_imu] if config else [7502, 7503]
    results = [listen(port, seconds) for port in ports]
    info("[result]")
    if not all(bound for bound, _, _ in results):
        info("    A port is taken by another program (see the owner above). Close it and run again.")
    elif results[0][1] == 0:
        info("    The ports are free but no packets arrive. Check in this order: operating_mode and udp_dest "
             "above, the cable and switch, then the firewall (allow python.exe on the private network).")
    else:
        info("    Packets arrive. The network is fine; the failure is inside the SDK, not the setup.")


def set_ports(host: str, lidar_port: int, imu_port: int, destination: str) -> None:
    core, sensor = _sdk()
    config = core.SensorConfig()
    config.udp_dest = destination
    config.udp_port_lidar = lidar_port
    config.udp_port_imu = imu_port
    sensor.set_config(host, config, persist=True, udp_dest_auto=False)
    info(f"sensor now sends to {destination}, lidar port {lidar_port}, imu port {imu_port}")
