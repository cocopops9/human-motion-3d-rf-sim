"""
Diagnose why no lidar frames arrive, one layer at a time.

    python check_sensor.py
    python check_sensor.py --host os-122542000054.local --seconds 4

Step 1  sensor configuration, read over HTTP (no UDP involved): where the
        sensor sends its data, on which ports, and whether it is streaming.
Step 2  plain UDP socket test without the Ouster SDK: can this program take
        ports 7502 and 7503, and do packets arrive on them?

Read the result:
    Step 2 cannot bind    another program owns the port (Ouster Studio, an old
                          Python process). Close it, or move the sensor to other
                          ports with --set-ports.
    Step 2 binds, 0 pkts  the sensor does not send to this PC: wrong udp_dest,
                          standby mode, cable or switch, or the Windows firewall.
    Step 2 binds, packets the network is fine and the problem is inside the SDK.

Only step 1 changes nothing on the sensor. --set-ports writes to the sensor.
"""

import argparse
import socket
import subprocess
import sys
import time

from ouster.sdk import core, sensor


def show_configuration(host):
    print(f"[1] sensor configuration ({host})")
    try:
        config = sensor.get_config(host)
    except Exception as error:
        print(f"    cannot read the configuration: {error}")
        print("    The sensor is not reachable over HTTP: check cable, switch, IP address.")
        return None

    print(f"    udp_dest        : {config.udp_dest}")
    print(f"    udp_port_lidar  : {config.udp_port_lidar}")
    print(f"    udp_port_imu    : {config.udp_port_imu}")
    print(f"    operating_mode  : {config.operating_mode}")
    print(f"    lidar_mode      : {config.lidar_mode}")
    print(f"    udp_profile     : {config.udp_profile_lidar}")

    if config.operating_mode == core.OperatingMode.STANDBY:
        print("    -> the sensor is in STANDBY and sends nothing.")
    if config.udp_dest is None or ":" in str(config.udp_dest):
        print("    -> udp_dest is missing or an IPv6 address; run setup_ip.py.")
    return config


def port_owner(port):
    """Windows only: which process owns a UDP port."""
    if not sys.platform.startswith("win"):
        return "unknown (not Windows)"
    command = (
        "Get-NetUDPEndpoint -LocalPort {p} -ErrorAction SilentlyContinue | "
        "ForEach-Object {{ $q = Get-Process -Id $_.OwningProcess -ErrorAction SilentlyContinue; "
        "'{{0}} (pid {{1}}) {{2}}' -f $q.ProcessName, $_.OwningProcess, $q.Path }}"
    ).format(p=port)
    try:
        result = subprocess.run(["powershell", "-NoProfile", "-Command", command],
                                capture_output=True, text=True, timeout=20)
        return result.stdout.strip() or "not listed (may need an administrator shell)"
    except Exception as error:
        return f"lookup failed: {error}"


def listen(port, seconds):
    """Bind a plain UDP socket and count what arrives. Returns (bound, packets, sources)."""
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        sock.bind(("0.0.0.0", port))
    except OSError as error:
        print(f"    port {port}: CANNOT BIND ({error})")
        print(f"      owner: {port_owner(port)}")
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
    print(f"    port {port}: bound, {packets} packets in {seconds:g} s"
          + (f" from {', '.join(sorted(sources))}" if sources else ""))
    return True, packets, sources


def test_sockets(config, seconds):
    print(f"[2] plain UDP socket test ({seconds:g} s per port, SDK not involved)")
    ports = [config.udp_port_lidar, config.udp_port_imu] if config else [7502, 7503]
    results = [listen(port, seconds) for port in ports]

    print("[result]")
    if not all(bound for bound, _, _ in results):
        print("    A port is taken by another program (see the owner above). Close it and rerun.")
    elif results[0][1] == 0:
        print("    The ports are free but no packets arrive. Check in this order: operating_mode and")
        print("    udp_dest above, the cable and switch, then Windows Defender Firewall (allow python.exe")
        print("    on the private network, or test once with the firewall off).")
    else:
        print("    Packets arrive. The network is fine; the failure is inside the SDK, not the setup.")


def set_ports(host, lidar_port, imu_port, destination):
    config = core.SensorConfig()
    config.udp_dest = destination
    config.udp_port_lidar = lidar_port
    config.udp_port_imu = imu_port
    sensor.set_config(host, config, persist=True, udp_dest_auto=False)
    print(f"sensor now sends to {destination}, lidar port {lidar_port}, imu port {imu_port}")


def main():
    parser = argparse.ArgumentParser(description="Diagnose missing Ouster lidar packets.")
    parser.add_argument("--host", default="os-122542000054.local")
    parser.add_argument("--seconds", type=float, default=3.0)
    parser.add_argument("--set-ports", type=int, nargs=2, metavar=("LIDAR", "IMU"),
                        help="write new UDP ports to the sensor, e.g. --set-ports 7602 7603")
    parser.add_argument("--dest", default="192.168.33.30", help="destination used with --set-ports")
    args = parser.parse_args()

    if args.set_ports:
        set_ports(args.host, args.set_ports[0], args.set_ports[1], args.dest)
        return

    config = show_configuration(args.host)
    test_sockets(config, args.seconds)


if __name__ == "__main__":
    main()
