.. _manual_parallel:

===============================
Parallel computation with PyHRF
===============================


.. Local
.. #####

LAN
###

Computation distribution uses *ssh* to launch commands on distant machines on a Local Aera Network (LAN) and retrieve results from them on the local machine. 

Setup
*****

The setup information to configure the parallel LAN feature of pyhrf is in the
section [parallel-LAN] of the file ~/.pyhrf/config.cfg::

    [parallel-LAN]
    niceness = 10 
    remote_path = /path/to/store/results
    hosts = localhost,localhost,localhost,localhost
    user = my_user
    remote_host = my_remote_host
    enable_unit_test = 1

where:

- "niceness" is the nice level with which command will be launched on distant 
  machines. The higher, the less priority the jobs will have (useful to not
  disturb other people that are using the distant machines as well).
- "remote_path" must be a path on the network that is accessible by all machines.
  It is used to store temporary results.
- "hosts" is a coma-separated list of machine hostnames on which the job will
  be launched. If one wants to launch multiple jobs on a given machine, 
  the hostname can be duplicated.
- "user" is a ssh login valid on all distant machines. 
- "remote_host" is a specific distant machine hostname that will be used to 
  retrieve results
- "enable_unit_test" is a flag to enable unit testing of the LAN feature

.. warning:: 
   The ssh connection on distant machines must be *non-interactive*. 
   To do so, the ssh key of the local user must be appended to the file 
   ``authorized_keys`` of the distant user. 

Test
****

To test the configuration prior to launching an actual analysis: 

 1. set the flag "enable_unit_test" to 1 in ``~/.pyhrf/config.cfg``
 2. run the following command::

      pyhrf_maketests pyhrf.test.test_treatment.TreatmentTest.test_default_treatment_parallel_LAN

Analysis
********

To launch a LAN-distributed analysis, simply add the option "-x LAN" to any ``pyhrf_..._estim`` command. For example, the default JDE analysis can be run with::

  pyhrf_jde_buildcfg          # build the XML parameter file
  pyhrf_jde_estim -v1 -x LAN  # launch the analysis using LAN-distributed computation




Cluster
#######

Configuration
*************

Setup
=====

To work on a cluster, pyhrf relies on `soma-workflow <http://brainvisa.info/soma/soma-workflow/>`_ which has to be installed on both sides (client/cluster). 
See the soma-workflow installation page for detailled instructions.

Launch a soma-workflow database server on the cluster::
    
    python -m soma.workflow.start_database_server SW_CLUSTER_ID &
    
Configure soma-workflow on client side, file ~/.soma-workflow.cfg::
 
    [SW_CLUSTER_ID]
    cluster_address = CLUSTER_HOSTNAME
    submitting_machines = CLUSTER_HOSTNAME
    
To ensure the communication from the server to the client, OpenSSH server must be installed and running::
    
    sudo aptitude install sshd
    sudo aptitude install openssh-server
    
Pyro must be installed to ensure the same configuration for the client and the server.    


Connexion with the server
=========================

To enable the connexion from the client to the server, the public key of the local user must be saved in the authorized keys of the server.
To enable the connexion from the server to the client, the public key of the server must be saved in the authorized keys of the local user.

.. image:: /figs/client_server.png.png

Configure the connexion with the server, file: ~/.pyhrf/config.cfg

Default content of the parallel-cluster section::
    
    [parallel-cluster]
    server_id = None
    server = None
    user = None
    remote_path = None
    
With the previously defined configuration, one should get::
    
    [parallel-cluster]
    server_id = SW_CLUSTER_ID
    server = CLUSTER_HOSTNAME
    user = CLUSTER_USER
    remote_path = CLUSTER_PYHRF_PATH
    
    
Launch a pyhrf analysis on the cluster
**************************************
In a directory containing the xml file (detectestim.xml), launch the analysis with::

    pyhrf_jde_estim -v1 -x cluster
  
Monitor execution (user interface to visualize running jobs, progression...)::
    
    soma_workflow_gui -u CLUSTER_USER -a
    
    
    
    

