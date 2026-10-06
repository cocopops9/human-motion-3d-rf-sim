function status = bodyscan(varargin)
%BODYSCAN Run a bodyscan command with the Python that MATLAB uses.
%
%   bodyscan('fuse', 'C:\lidar\tt17', '--out', 'C:\lidar\person_tt17')
%   bodyscan('mesh', 'C:\lidar\person_tt17.ply', '--out', 'C:\lidar\person_tt17_mesh.ply')
%   bodyscan('detect', 'C:\lidar\tt17', '--rotation', '--human')
%   bodyscan('--help')
%
%   The Python interpreter is the one configured in MATLAB (pyenv), so the
%   packages installed for it (numpy, open3d, ouster-sdk) are used. The
%   repository folder is the parent of the folder of this file; it is put on
%   PYTHONPATH for the command. Returns the exit status of the command
%   (0 = success); the output is printed while the command runs.
%
%   Every option of a command: bodyscan('<command>', '--help').

    environment = pyenv;
    if strlength(environment.Executable) == 0
        error('bodyscan:python', ['No Python configured in MATLAB: set it once with ', ...
              'pyenv(''Version'', ''C:\path\to\python.exe'')']);
    end
    repository = fileparts(fileparts(mfilename('fullpath')));
    arguments = cellfun(@quote, varargin, 'UniformOutput', false);
    command = sprintf('"%s" -m bodyscan %s', environment.Executable, strjoin(arguments, ' '));
    previous = getenv('PYTHONPATH');
    if isempty(previous)
        setenv('PYTHONPATH', repository);
    else
        setenv('PYTHONPATH', [repository, pathsep, previous]);
    end
    cleanup = onCleanup(@() setenv('PYTHONPATH', previous));
    status = system(command, '-echo');
end

function text = quote(argument)
    text = char(string(argument));
    if any(text == ' ') || any(text == '"')
        text = ['"', strrep(text, '"', '\"'), '"'];
    end
end
