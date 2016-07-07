.. _pyhrf_configuration:

Configuration
#############

Package options are stored in ``$HOME/.pyhrf/config.cfg``, which is created after the installation. It handles global package options and the setup of parallel processing. Here is the default content of this file (section order may change)::


    [global]
    write_texture_minf = False          ; compatibility with Anatomist for texture file
    tmp_prefix = pyhrftmp               ; prefix used for temporary folders in tmp_path
    verbosity = 0                       ; default of verbosity, can be changed with option -v
    tmp_path = /tmp/                    ; where to write file
    use_mode = enduser                  ; "enduser": stable features only, "devel": +indev features
    spm_path = None                     ; path to the SPM matlab toolbox (indev feature)


    [parallel-cluster]                  ; Distributed computation on a cluster.
                                        ; Soma-workflow is required.
                                        ; Authentification by ssh keys must be
                                        ; configured in both ways (remote <-> local)
                                        ; -> eg copy content of ~/.ssh/id_rsa.pub (local machine)
                                        ;    at the end of ~/.ssh/authorized_keys (remote machine)
                                        ;    Also do the converse:
                                        ;    copy content of ~/.ssh/id_rsa.pub (remote machine)
                                        ;    at the end of ~/.ssh/authorized_keys (local machine)

    server_id = None                    ; ID of the soma-workflow-engine server
    server = None                       ; hostname or IP adress of the server
    user = None                         ; user name to log in the server
    remote_path = None                  ; path on the server where data will be stored

    [parallel-local]                    ; distributed computation on the local cpu
    niceness = 10                       ; niceness of remote jobs
    nb_procs = 1                        ; number of distruted jobs, better not over
                                        ; the total number of CPU
                                        ; 'cat /proc/cpuinfo | grep processor | wc -l' on linux
                                        ; 'sysctl hw.ncpu' on MAC OS X
                                        ; Set it to 0 if you want to auto-detect the number of
                                        ; availables cpus and cores (taking into account the kernel
                                        ; restrictions such as cgroups, ulimit, ...)

    [parallel-LAN]                      ; Distributed computation on a LAN
                                        ; Authentification by ssh keys must be
                                        ; configured
    remote_host = None                  ; hostname or IP address of a host on the LAN
    niceness = 10                       ; niceness for distributed jobs
    hosts = $HOME/.pyhrf/hosts_LAN      ; plain text file containing coma-separated list of hostnames on the LAN
    user = None                         ; user name used to log in on any machine
                                        ; on the LAN
    remote_path = None                  ; path readable from the machine where
                                        ; pyhrf is launched (to directly retrieve
                                        ; results)

.. see :ref:`Parallel Computation <manual_parallel>`
