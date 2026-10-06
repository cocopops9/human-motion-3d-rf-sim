% girogirotondo_timer.m
% Rotate the platform by turn_deg degrees after a countdown, so that the
% person has time to start the LiDAR recording on the other PC and to step
% onto the platform.
%
% Same motor configuration and serial command as girogirotondo.m. Run it
% FIRST, then start 'python -m bodyscan capture-turntable' on the LiDAR PC 5 to 30 s later
% (for delay_s = 60). It beeps once per second in the last 10 s (higher in
% the last 3) and prints the time and duration of every move: send that
% printout along with the recording.
%
% Any rotation works for 'bodyscan fuse': a partial turn (e.g. 50 deg), one
% lap (360) or several laps (e.g. 1800 = 5 laps; more laps give more views of
% every part of the body). A rotation longer than max_steps_per_command is
% sent as several commands in a row (one lap each by default) with a short
% stop in between: 28800 steps (one lap) is known to work, while a larger
% number may not fit the integer type of the controller firmware (a 16-bit
% int ends at 32767). 'bodyscan fuse' handles both, but one continuous
% move gives better angles (no stops to model). To test the firmware, with
% the platform EMPTY run once
%     motor_ctrl.moveTurntableSteps(144000)
% after creating motor_ctrl: if the platform turns exactly 5 laps, set
% max_steps_per_command = Inf. If it turns about 162 deg instead (144000
% truncated to 16 bits), or does anything else, keep 28800.
%
% The LiDAR recording (capture-turntable --duration, or --turn-deg to compute it) must last until a
% few seconds after the platform stops: at least
%     (delay_s - time between starting this script and the LiDAR script - 18 s)
%     + rotation time + 5 s
% The time of every move is printed: N laps take about N times one lap,
% plus about 1 s per stop between commands.

clear all;
fprintf('girogirotondo_timer version 2026-10-01c (direct serial, no library timeout, frees COM port)\n');

delay_s = 60;                    % seconds from now until the platform starts turning
turn_deg = 1440;                 % total rotation [deg]: 50, 360, 1800, ...
max_steps_per_command = 28800;   % one lap per serial command (Inf = one continuous move)

%% configuration of the movement control (as in girogirotondo.m)
motors_comport = "COM3";
motors_boudrate = 9600;
camera_turntable = "192.168.33.12";
camera_rail = "192.168.33.13";
roi_turntable =  [248 200 985 63];
roi_rail =  [218 304 884 145];
total_codes_turntable = 3552;
codes_per_mm_rail = 1;
motors_step_per_rev = 800;
turntable_gear_ratio = 36;
motor_steps_per_radiant = (motors_step_per_rev * turntable_gear_ratio)/(2*pi);
rail_gear_ratio = 129.645;
motor_steps_per_mm = motors_step_per_rev/rail_gear_ratio;
angle_tollerance = 0.3*pi/180;
position_tollerance = 2.5;
num_attempts = 30;
railmin = 150;
railmax = 1300;
timeout = seconds(1800); % per command (long enough for many laps in one command); a duration, as wait_turntable expects

% A run that crashed leaves its serial port open (the port's callback keeps
% the old controller alive, so "clear all" does not release it): close every
% serialport object MATLAB still holds before opening COM3 again.
if exist('serialportfind', 'file')
    stale_ports = serialportfind;
    if ~isempty(stale_ports)
        fprintf('Closing %d serial port(s) left open by a previous run\n', numel(stale_ports));
        delete(stale_ports);
    end
end

% The controller (and its cameras) are set up BEFORE the countdown starts,
% so the countdown is exact.
motor_ctrl = movement_control(motors_comport, motors_boudrate, ...
                              camera_turntable, camera_rail, ...
                              roi_turntable, roi_rail, ...
                              total_codes_turntable, codes_per_mm_rail, ...
                              motor_steps_per_radiant, motor_steps_per_mm, ...
                              angle_tollerance, position_tollerance, ...
                              num_attempts, railmin, railmax, timeout, ...
                              "", 0);

%% split the rotation into commands
total_steps = round(motor_ctrl.motor_steps_per_radiant_turntable * turn_deg * pi / 180);   % 80 steps per degree
commands = [];
remaining_steps = total_steps;
while remaining_steps > 0
    chunk = min(remaining_steps, max_steps_per_command);
    commands(end + 1) = chunk; %#ok<SAGROW>
    remaining_steps = remaining_steps - chunk;
end
fprintf('Rotation: %g deg = %d steps in %d command(s)\n', turn_deg, total_steps, numel(commands));

%% countdown
fs = 8000;
tone = @(f, d) sound(0.4 * sin(2*pi*f*(0:1/fs:d)), fs);
fprintf('Platform starts in %d s. Start python -m bodyscan capture-turntable now.\n', delay_s);
t0 = tic;
last_announced = -1;
while toc(t0) < delay_s
    remaining = ceil(delay_s - toc(t0));
    if remaining ~= last_announced
        last_announced = remaining;
        if remaining <= 3
            tone(1500, 0.15);
        elseif remaining <= 10
            tone(1000, 0.12);
        end
        if remaining <= 10 || mod(remaining, 10) == 0
            fprintf('  %d\n', remaining);
        end
    end
    pause(0.05);
end

%% rotation
start_time = datetime("now");
tone(2000, 0.6);
fprintf('Turning since %s\n', char(start_time, 'HH:mm:ss.SSS'));
% The command is sent here directly instead of through moveTurntableSteps:
% its internal timeout stopped a 4-lap run after the first lap. Here the
% script waits for the controller's "T done" (which clears
% motor_ctrl.turntable_moving through the serial callback) for as long as
% the move needs. Only if "T done" never arrives does it give up, after
% max_wait_per_lap_s per lap, and send the next command anyway, so the
% platform always receives the whole rotation.
max_wait_per_lap_s = 600;
for k = 1:numel(commands)
    move_start = datetime("now");
    motor_ctrl.turntable_moving = 1;
    writeline(motor_ctrl.serial_port, ['[T-' num2str(commands(k)) ']']);   % '-' = positive steps, as in movement_control.m
    max_wait_s = max(60, max_wait_per_lap_s * commands(k) / 28800);
    while motor_ctrl.turntable_moving && seconds(datetime("now") - move_start) < max_wait_s
        pause(0.05);
    end
    if motor_ctrl.turntable_moving
        fprintf('  move %d/%d: no "T done" after %.0f s, sending the next command anyway\n', ...
                k, numel(commands), max_wait_s);
        motor_ctrl.turntable_moving = 0;
    end
    pause(0.25);
    move_s = seconds(datetime("now") - move_start) - 0.25;
    fprintf('  move %d/%d: %d steps (%.1f deg) in %.2f s, average %.2f deg/s\n', ...
            k, numel(commands), commands(k), commands(k) / 80, move_s, commands(k) / 80 / move_s);
end
end_time = datetime("now");
turn_s = seconds(end_time - start_time) - 0.25;
fprintf('Rotation finished at %s: %.1f deg in %.2f s, average %.2f deg/s including stops\n', ...
        char(end_time, 'HH:mm:ss.SSS'), turn_deg, turn_s, turn_deg / turn_s);
tone(1500, 0.2); pause(0.3); tone(1500, 0.2);

delete(motor_ctrl.serial_port);   % release COM3 explicitly (the callback would keep it open)
clear motor_ctrl;
