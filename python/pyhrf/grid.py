#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Module to distribute shell commands across a network

Original Author: Mathieu Perrot
Extended by: Thomas Vincent
"""

import os, sys, numpy, re, getpass, time, datetime, signal
import exceptions, warnings
import socket, threading
from optparse import OptionParser
warnings.filterwarnings('ignore', 'Python C API version mismatch',
                                                RuntimeWarning)
try:
    import paramiko
except ImportError:
    pass

from pyhrf.tools.message import msg

# Commentaires divers :
# ---------------------
# - Ce script suppose que l'hote distant est un linux, kernel 2.6. A tester avec
# kernel 2.4...pas sur que ca marche : a cause du contenu de /proc.
# - Ce script utilise le fait que le shell distant est un shell bash,
# il faudrait tester avec d'autre shell pour voir s'il marche avec sh.
# - il utilise les commandes distantes suivantes :
#   * cat, sed, grep, tty
# - il utilise les builtins du shell suivantes :
#   * ulimit


# FIXME
# 1) ajouter aide detaillee pour --user, --password
# pour --password : 'GET', '', '-' : pour rentrer le mot de passe a la main
# 2) mode tache hierarchique :
#    - erreur si 2 fois la meme regle de defini
#    - erreur si boucle : au runtime c'est plus facile
#    - je crois que seul les arbres marche.


def remote_dir_is_writable(user, hosts, path):
    """
    Test if *path* is writable from each host in *hosts*. Sending bash
    commands to each host via ssh using the given *user* login.

    Args:

    """
    import os.path as op
    import pyhrf

    mode = 'dispatch'
    cmds = ['bash -c "echo -n \"[write_test]:%d:\"; '\
            'if [ -w %s ]; then echo \"OK\"; else echo \"NOTOK\"; fi;"' \
            %(ih, path) for ih in range(len(hosts))]
    tasks = read_tasks(cmds, mode)
    timeslot = read_timeslot('allday')

    tmp_dir = pyhrf.get_tmp_path()
    logfile = op.join(tmp_dir, 'pyhrf.log')
    run_grid(mode, hosts, 'rsa', tasks, timeslot, logfile=logfile, user=user)
    kill_threads()
    log = open(logfile).readlines()

    res = [False] * len(hosts)
    for line in log:
        if line.startswith('[write_test]'):
            #print line
            _,ih,r = line.strip('\n').split(':')
            res[int(ih)] = ('OK' in r)
    os.remove(logfile)
    return res

def create_options(argv):
    cmd = os.path.basename(argv[0])
    #description = 'Grid computing'
    usage = '''Grid computing
%s --hosts HOSTS --tasks TASKS [options]
   Dispatch TASKS list on HOSTS list.
%s --hosts HOSTS [options]
   Only probe hosts.''' % (cmd, ''.ljust(len(cmd)))
    parser = OptionParser(usage = usage)
    parser.add_option('--hosts', dest='hosts',
        metavar='HOSTS', action='store', default=None,
        help='hosts file or hosts list ("host1, host2, host3")')
    parser.add_option('--morehelp', dest='morehelp',
        metavar='OPT', action='store', default=None,
        help='details help on OPT (ex : --morehelp hosts)')
    parser.add_option('--tasks', dest='tasks',
        metavar='FILE', action='store', default=None,
        help='tasks file (one task by line)')
    parser.add_option('--mode', dest='mode',
        metavar='MODE', action='store', default='dispatch',
        help='task manager mode : dispatch,repeat,hierarchical/hie')
    parser.add_option('--log', dest='logfile',
        metavar='FILE', action='store', default="grid.log",
        help='log output(stdout/stderr) of commands (default %default)')
    parser.add_option('--broken', dest='brokenfile',
        metavar='FILE', action='store', default="broken_cmd.batch",
        help='file to store commands returning bad exist status '
            '(default %default)')
    parser.add_option('--user', dest='user',
        metavar='USER', action='store', default=None,
        help='set user name (default $USER)')
    parser.add_option('--passwd', dest='passwd',
        metavar='PASSWD', action='store', default=None,
        help='set user password (default : no password, use ssh key)')
    parser.add_option('--keytype', dest='keytype',
        metavar='TYPE', action='store', default='dsa',
        help='dsa or rsa (key type)')
    parser.add_option('--timeslot', dest='timeslot',
        metavar='T', action='store', default='night',
        help='define timeslot when tasks can be started (default : night)')
    return parser

def log_help(cmd):
    print '''
--------------------------------------------------------------------------------
 --log FILE (FILE default value = grid.log)

    Store standard output and error output of each task, logging starting date,
    endding date, computing time, computing host and associated command line.
--------------------------------------------------------------------------------
'''

def broken_help(cmd):
    print '''
--------------------------------------------------------------------------------
 --broken FILE (FILE default value = broken_cmd.batch)

    Store all broken commands (those which returns bad exist status, i.e
    different from 0 : see echo $? on bash shell) on FILE.

    $ cat FILE
    ls ~/plop
    $ ls ~/plop
    ls : /home/???/plop: No such file or directory
    $ echo $?
    1
--------------------------------------------------------------------------------
'''

def mode_help(cmd):
    print '''
--------------------------------------------------------------------------------
 --mode MODE
    MODE in [ dispatch, repeat ]

 * dispatch :   each task is dispatch on computers from hosts list (see --hosts)

 * repeat :     repeat CMD or SCRIPT tasks (see --tasks)

 * hierarchic : handle hierarchic tasks from file
   File format :                   Task dependencies :

   all: a b FILE                           all
   a:                                      / \\
     echo "task a"                        a   b
   b:
     echo "task b"


   Here FILE follows hierarchical task format or a classical task format.

--------------------------------------------------------------------------------
'''

def tasks_help(cmd):
    print '''
--------------------------------------------------------------------------------
 * Several tasks dispatch on several hosts
 --tasks FILE
    $ cat FILE
    sh cmd1.sh --opt1
    python cmd2.py --opt2 arg2
    $ %s --tasks FILE

 * One task repeated on each hosts
 --tasks SCRIPT --mode repeat
    $ cat SCRIPT
    #!/usr/bin/env bash
    echo "my script on $HOSTNAME";
    cat /proc/cpuinfo;

 * Several tasks dispatch on several hosts
 --tasks CMD
    $ %s --tasks 'sh cmd1.sh; python cmd2.py'

 * One task repeated on each hosts
 --tasks CMD --mode repeat
    $ %s --tasks 'echo $HOSTNAME'

 * Empty task (or empty file task) : only probe hosts
 --tasks ''
--------------------------------------------------------------------------------
''' % (cmd, cmd, cmd)

def hosts_help(cmd):
    print '''
--------------------------------------------------------------------------------
* Several hosts from a file
 --hosts FILE
    $ cat file
    file1 file2 file3
    $ %s --hosts file

* Several hosts from command line
 --hosts HOSTS_LIST
    $ %s --hosts host1,host2,host3

* One host from command line
    $ %s --hosts host1,
--------------------------------------------------------------------------------
''' % (cmd, cmd, cmd)
    sys.exit(1)

def timeslot_help(cmd):
    print '''
--------------------------------------------------------------------------------
 * Tasks are started beetween 2 times
 --timeslot H1:M1:S1-H2:M2:S2

 * Tasks are started before one time
 --timeslot -H

 * Tasks are started after one time
 --timeslot H-

 * Complex timeslots (list of timeslots)
 --timeslot -H1,H2-H3,H4-

 * No restriction
 --timeslot allday
 --timeslot -
 --timeslot 0:00:00,24:00:00

 * Shortcuts for current timeslots :
 --timeslot night :         00:00:00-08:00:00,20:00:00-24:00:00
 --timeslot midi :          12:00:00-14:00:00
 --timeslot allday :        00:00:00-24:00:00
 --timeslot morning :       08:00:00-12:00:00
 --timeslot afternoon :     12:00:00-20:00:00
--------------------------------------------------------------------------------
'''



def read_hosts(hosts):
    if hosts.find(',') == -1:
        try:
            fd = open(hosts, 'r')
        except exceptions.IOError:
            msg.error("host file '%s' can not be opened" % hosts)
        else:
            hosts = ' '.join(fd.readlines()).replace('\n', ' ')
            fd.close()
    hosts_list = re.split('[\s,]+', hosts)
    if hosts_list[-1] == '': del hosts_list[-1]
    return hosts_list


def read_hierarchic_tasks(tasks_file):
    def read_file_list(file):
        fd = open(file, 'r')
        lines = fd.readlines()
        tasks = TaskList(lines)
        fd.close()
        return tasks

    def read_deps_tokens(line):
        # task : dep1 dep2 dep3
        taskre = re.compile('[ \t]*:[ \t]*')
        try:
            task, deps = re.split(taskre, line)
        except ValueError:
            return None
        deps = deps.rstrip('\n')
        depsre = re.compile('[\t ]+')
        deps = re.split(depsre, deps)
        if deps == ['']: deps = []
        return task, deps

    def read_file_hier(file, tasks_dic):
        DEPS_STATE, TASKS_STATE = 0, 1
        fd = open(file, 'r')
        lines = fd.readlines()
        state = DEPS_STATE
        tasks = []
        task = None
        deps = None
        for l in lines:
            blankre = re.compile('^[ \t]*$')
            if blankre.match(l): continue
            depsre = re.compile('[a-zA-Z0-9_]*[ \t]*:[ \t]*')
            if depsre.match(l):
                state = DEPS_STATE
            if state == DEPS_STATE:
                if task: tasks_dic[task] = deps, tasks
                tokens = read_deps_tokens(l)
                state = TASKS_STATE
                tasks = []
                if tokens is None:
                    task, deps = "all", []
                    tasks.append(l)
                else:    task, deps = tokens
            elif state == TASKS_STATE:
                tasks.append(l)
        if task is not None: tasks_dic[task] = deps, tasks
        fd.close()

    def create_hietask(file):
        tasks_dic = {}
        read_file_hier(file, tasks_dic)
        if len(tasks_dic) == 0:
            msg.error("task file is empty")
            sys.exit(1)
        for v, (deps, tasks) in tasks_dic.items():
            for d in deps:
                if tasks_dic.has_key(d): continue
                try:
                    hietask = create_hietask(d)
                    tasks_dic[d] = hietask
                except exceptions.IOError:
                    e = "in file '" + file + \
                        "', rule or file '" + d + \
                        "' does not exist"
                    msg.error(e)
                    sys.exit(1)
        return TaskHierarchical(file, tasks_dic)

    try:
        return create_hietask(tasks_file)
    except exceptions.IOError:
        msg.error("file '%s' does not exist.")
        sys.exit(1)


def read_tasks(tasks, mode):
    if mode in ['hierarchical', 'hie']:
        return read_hierarchic_tasks(tasks)
    if mode == 'dispatch' and isinstance(tasks, list):
        rtasks = TaskList(tasks)
    else:
        try:
            # SCRIPT / BATCH
            fd = open(tasks, 'r')
            lines = fd.readlines()
            if mode == 'dispatch': rtasks = TaskList(lines)
            elif mode == 'repeat': rtasks = Task(';'.join(lines))
            fd.close()
        except exceptions.IOError:
            if mode == 'dispatch': rtasks = TaskList(tasks.split(';'))
            elif mode == 'repeat' : rtasks = Task(tasks)
    return rtasks


def read_timeslot(timeslot):
    if timeslot in ( '-', 'allday' ):
        return TimeSlot(0, 86400) # [0:24h]
    elif timeslot == 'morning':
        return TimeSlot(0, 43200) # [0:12h]
    elif timeslot == 'afternoon':
        return TimeSlot(43200, 86400) # [12h:24h]
    elif timeslot == 'midi':
        return TimeSlot(43200, 50400) # [12h:14h]
    elif timeslot == 'night':
        # [0:8h] + [20h:24h] = [20h:8h]
        return TimeSlotList([TimeSlot(0, 28800),
                TimeSlot(72000, 86400)])

        timeslot_list = timeslot.split(',')

    tsl = []
    for one_timeslot in timeslot_list:
        try:
            t1, t2 = one_timeslot.split('-')
        except ValueError:
            msg.error("'%s' : bad timeslot format" % one_timeslot)
            sys.exit(1)
        seconds = []
        for t in [t1, t2]:
            try:
                h, m, s = t.split(':')
            except ValueError:
                msg.error("'%s' : bad hour format" % t)
                sys.exit(1)
            h, m, s = int(h), int(m), int(s)
            seconds.append(s + 60 * (m + 60 * h))
        tsl.append(TimeSlot(seconds[0], seconds[1]))
    if len(timeslot_list) > 1:
        ts = TimeSlotList(tsl)
    else:    ts = tsl[0]
    return ts

def parse_options(parser):
    morehelpmap = {'hosts' : hosts_help, 'tasks' : tasks_help,
        'mode' : mode_help, 'broken' : broken_help, 'log' : log_help,
        'timeslot' : timeslot_help}
    (options, args) = parser.parse_args(sys.argv)
    error = False
    error_msg = []

    if options.morehelp != None:
        try:
            morehelpmap[options.morehelp](sys.argv[0])
            sys.exit(1)
        except KeyError:
            msg.error("No help for '%s'.\n" % options.morehelp)
            msg.write_list([' ', ('*', 'green'),
                ' List of available full helps :\n   '])
            for k in morehelpmap.keys(): msg.write('%s, ' % k)
            msg.write('\n')
            sys.exit(1)
    mandatory_options = {'hosts' : options.hosts}
    timeslot = read_timeslot(options.timeslot)
    if None in mandatory_options.values():
        opt = [n for n, o in mandatory_options.items() if o == None]
        error_msg.append('missing options : %s' % opt)
        error = True
    if not options.mode in ["dispatch", "repeat", "hie", "hierarchical"]:
        error_msg.append("'%s' : unknown task manager mode." % \
                            options.mode)
        error = True
    if error:
        parser.print_help()
        for m in error_msg: msg.error(m)
        sys.exit(1)
    hosts_list = read_hosts(options.hosts)
    if options.tasks:
        tasks_list = read_tasks(options.tasks, options.mode)
    else:    tasks_list = None
    return options, timeslot, hosts_list, tasks_list


class TimeSlot(object):
    '''
    TimeSlot(start, end)

    Define a contiguous timeslot.

    start, end : in second since day beggining.
    '''
    def __init__(self, start, end):
        if start == '':
            self._start = 0
        else:    self._start = start
        if end == '':
            self._end = 86400 # 24h in seconds
        else:    self._end = end

    def is_inside(self, time):
        return time > self._start and time < self._end

    def is_inside_now(self):
        now = numpy.array(time.localtime())
        h, m, s = now[3:6]
        t = s + 60 * (m + 60 * h)
        return self.is_inside(t)


class TimeSlotList(TimeSlot):
    '''
    TimeSlotList(list)

    Define uncontiguous timslots.

    list :  list of timeslots.
    '''
    def __init__(self, list):
        self._list = list

    def is_inside(self, time):
        a = numpy.array([l.is_inside(time) for l in self._list])
        return a.any()


class Task(object):
    '''
    Only one task that will be computed on only one host.
    '''
    def __init__(self, task):
        self._task = task

    def __len__(self):
        return 1

    def get(self):
        return self._task

    def __repr__(self):
        return self._task

class TaskList(Task):
    '''
    List of independent tasks. Each one can be computed on a different task.
    '''
    def __init__(self, tasks):
        self._tasks = tasks

    def __len__(self):
        return len(self._tasks)

    def next(self):
        t = self._tasks.pop()
        return Task(t)

    def append(self, task):
        self._tasks.append(task)

        def __repr__(self):
                return str(self._tasks)

    def __repr__(self):
        return str(self._tasks)


class TaskHierarchical(Task):
    '''
    Hiearchic dependencies of TaskList.
    '''
#FIXME need more doc
    def __init__(self, rule, tasks_dic):
        self.rule = rule
        self._tasks = tasks_dic

    def init(self):
        self._deps_stack = [0]
        self._rule_stack = ['all']
        msg.write_list(['- ', ('all', 'blue'), ' : %s\n' % \
                ', '.join(self._tasks['all'][0])])

    def __len__(self):
        a = numpy.array([len(t) for t in self._tasks.values()])
        return a.sum()

    def _find_leaf(self):
        if len(self._rule_stack) == 0: return True
        rule = self._rule_stack[-1]
        deps, tasks = self._tasks[rule]
        if deps:
            subrule = deps[self._deps_stack[-1]]
            val = self._tasks[subrule]
            self._rule_stack.append(subrule)
            self._deps_stack.append(0)
            if isinstance(val, TaskHierarchical):
                return False
            else: subdeps, subtasks = val
            msg.write_list(['  ' * len(self._rule_stack), '- ',
                ('%s' % subrule, 'blue'),
                ' : %s\n' % ', '.join(subdeps)])
            return self._find_leaf()
        else:    return True

    def _go_up_for_next_leaf(self):
        while 1:
            if not len(self._rule_stack): break
            rule = self._rule_stack[-1]
            val = self._tasks[rule]
            if isinstance(val, TaskHierarchical):
                self._rule_stack.pop()
                self._deps_stack.pop()
                continue
            deps, tasks = val
            dep_number = self._deps_stack[-1]
            if dep_number >= len(deps):
                if tasks == [] or dep_number > len(deps):
                    self._rule_stack.pop()
                    self._deps_stack.pop()
                    continue
            else:
                self._deps_stack[-1] += 1
                break

    def _pop_subhie(self):
        rule = self._rule_stack[-1]
        self._deps_stack[-1] += 1
        return self._tasks[rule]

    def next(self):
        rule = self._rule_stack[-1]
        val = self._tasks[rule]
        if isinstance(val, TaskHierarchical): val = [], []
        deps, tasks = val

        if self._deps_stack[-1] == 0 and deps != []:
            if not self._find_leaf(): return self._pop_subhie()
            deps, tasks = self._tasks[rule]

        # if at the end of deps :
        if self._deps_stack[-1] >= len(deps):
            # no task or task done -> go up
            if tasks == [] or self._deps_stack[-1] > len(deps):
                self._go_up_for_next_leaf()
                if self._rule_stack == []: return None
                rule = self._rule_stack[-1]
                deps, tasks = self._tasks[rule]
                if self._deps_stack[-1] < len(deps):
                    if not self._find_leaf():
                        return self._pop_subhie()

                elif tasks != []:
                    msg.write_list(['  ' * \
                        len(self._rule_stack), '<- ',
                        ('%s' % rule, 'blue'), '\n'])
                rule = self._rule_stack[-1]
                deps, tasks = self._tasks[rule]
        self._deps_stack[-1] += 1
        return TaskList(list(tasks))


class User(object):
    """Define user launching task and identification process.

    Parameters
    ----------
    name
        username.
    passwd
        user passwd or if None, try to get `~/.ssh/id_dsa` dsa key for key connection.
    keytype
        `rsa` or `dsa`.
    """
    def __init__(self, name=None, passwd=None, keytype=None):
        if name is None:
            name = os.getenv('USER')
        self.name = name
        self.keytype = keytype
        if passwd is None:
            self.passwd = None
            if self.keytype == 'dsa' :
                self.keyfile = os.path.join(
                    os.getenv('HOME'), '.ssh', 'id_dsa')
            elif self.keytype == 'rsa' :
                self.keyfile = os.path.join(
                    os.getenv('HOME'), '.ssh', 'id_rsa')
            else:    self.keyfile = None
        elif passwd in ['GET', '', '-']:
            self.keyfile = None
            self.passwd = getpass.getpass()
        else:
            self.keyfile = None
            self.passwd = passwd

    def key(self):
        if self.keyfile is None: return None
        elif self.keytype == 'dsa':
            key = paramiko.DSSKey(filename=self.keyfile)
        elif self.keytype == 'rsa':
            key = paramiko.RSAKey(filename=self.keyfile)
        return key


class Host(object):
    '''
    Host(name, status)

    name :   hostname.
    status : set host status :
    '''
    #FIXME : add host status list
    def __init__(self, name, status):
        self._name = name
        self._status = status

    def __repr__(self):
        return self._name


class ProbeHost(threading.Thread):
    def __init__(self, hosts_manager, hostname):
        threading.Thread.__init__(self)
        self._hosts_manager = hosts_manager
        self._hostname = hostname

    def run(self):
        up, status = self._hosts_manager.isup(self._hostname)
        host = Host(self._hostname, status)
        self._hosts_manager.update_host_status(host, status)
        self._hosts_manager._hosts_probed_number += 1


class HostsManager(object):
    available_status = 0
    not_available_status = 1
    unknown_host_status = 2
    unknown_status = 3

    def __init__(self, list):
        self._list = list
        self._init_lists()
        self.update_all_hosts()

    def _init_lists(self):
        self._available_list = []
        self._not_available_list = []
        self._unknown_list = []
        self._unknown_status_list = []
        HM = HostsManager
        self.status_lists = {\
            HM.available_status : self._available_list,
            HM.not_available_status : self._not_available_list,
            HM.unknown_host_status : self._unknown_list,
            HM.unknown_status : self._unknown_status_list }

    def update_all_hosts(self):
        self._init_lists()
        self._hosts_probed_number = 0
        for h in self._list:
            ph = ProbeHost(self, h)
            ph.start()
        while 1:
            if self._hosts_probed_number == len(self._list): break
            time.sleep(0.01)
        hosts_lists = {
            'available hosts' :     self._available_list,
            'unavailable hosts' : self._not_available_list,
            'unknown hosts' :       self._unknown_list,
            'unknown status' :      self._unknown_status_list}
        for name, list in hosts_lists.items():
            if len(list) == 0: continue
            msg.write_list([(' * ', 'bold_yellow'), '%s : %s' % \
                            (name, ' ' * (20 - len(name))),
                            ('%d' % len(list), 'cyan'), " ",
                            str(list), '\n'])

    def update_host_status(self, host, status):
        for l in self.status_lists.values():
            try: l.remove(host)
            except ValueError: pass
        self.status_lists[status].append(host)

    def probe(self, hostname):
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(3.)
            sock.connect((hostname, 22))
            t = paramiko.Transport(sock)
            return t, self.available_status
        except socket.error, e:
            #print 'hostname', hostname
            #print e
            return None, self.not_available_status
        except socket.gaierror:
            return None, self.unknown_host_status

    def isup(self, hostname):
        t, status = self.probe(hostname)
        if t:
            t.close()
            return True, status
        else:    return False, status


class TasksStarter(threading.Thread):
    def __init__(self, tasks_manager, host, task, time_limit=86400):
        threading.Thread.__init__(self)
        self._tasks_manager = tasks_manager
        self._host = host
        self._task = task
        self._lock = threading.Lock()
        tasks_manager._active_hosts.append(self._host)
        self._killed = False
        self._pty = None
        self._stdout_data = None
        self._start_time = None
        self._status = None
        self._time_limit = time_limit

    def _stop(self, transport, channel):
        if channel: channel.close()
        if transport: transport.close()
        try:
            self._tasks_manager._active_hosts.remove(self._host)
        # task allready finished
        except ValueError: pass

    def _bad_stop(self, transport, channel):
        self._lock.acquire()
        self._stop(transport, channel)
        if not self._killed:
            msg.warning("remove host '%s' from avalaible hosts." % \
                                self._host)
            self._tasks_manager.abnormal_stop(self._task.get())
        self._lock.release()

    def _remote_killed_stop(self, transport, channel):
        self._write_killed_log()
        self._bad_stop(transport, channel)

    def _log_header(self):
        start_time = self._start_time
        stop_time = time.time()
        diff_time = datetime.timedelta(0, int(stop_time - start_time))
        header = "# %s : %s -> %s, on %s\n# %s\n" % (diff_time, \
            time.asctime(time.localtime(start_time)),
            time.asctime(time.localtime(stop_time)),
            self._host, self._task.get().rstrip('\n'))
        header += '# exit status = %s\n' % self._status
        return header

    def _write_killed_log(self):
        header = self._log_header()
        tm = self._tasks_manager
        self._lock.acquire()
        s = self._status - 128
        error = "grid.py error : Killed process by signal %d." % s
        tm._log.write(header + '\n' + error + '\n')
        tm._log.flush()
        self._lock.release()

    def _write_log(self, channel):
        tm = self._tasks_manager
        stdout = [l.strip('\n') for l in channel.makefile('r+')]
        if len(stdout) == 0:
            stdout = [self._stdout_data]
        else:    stdout[0] = self._stdout_data + stdout[0]
        stderr = [l.strip('\n') for l in channel.makefile_stderr('r+')]

        if len(stderr):
            stderr.reverse(); stderr.append('# (stderr)')
            stderr.reverse();
        header = self._log_header()
        self._lock.acquire()
        tm._log.write('\n'.join([header] + stdout + stderr + ['\n']))
        tm._log.flush()
        self._lock.release()

    def _write_broken(self):
        self._lock.acquire()
        tm = self._tasks_manager
        tm._brokenfd.write(self._task.get().strip() + '\n')
        tm._brokenfd.flush()
        self._lock.release()

    def _good_stop(self, transport, channel):
        self._write_log(channel)
        if self._status != 0: self._write_broken()
        self._lock.acquire()
        self._stop(transport, channel)
        self._tasks_manager._free_hosts.append(self._host)
        self._tasks_manager._tasks_finished_number += 1
        self._lock.release()

    def _read_pty(self, chan):
        data = chan.recv(20)
        ind = data.find(' ')
        if ind == -1: ind = len(data)
        self._pty, self._stdout_data = data[:ind], data[ind + 1:]

    def _get_connection_params(self):
        tm = self._tasks_manager
        username = tm._user.name
        key = tm._user.key()
        if key is not None:
            params = {'username' : username, 'pkey' : key}
        else:
            passwd = tm._user.passwd
            params = {'username' : username, 'password' : passwd}
        return params

    def kill(self):
        tm = self._tasks_manager
        try:
            t, status = tm._hosts_manager.probe(self._host._name)
            t.connect(**self._get_connection_params())
            chan = t.open_session()
            cmd = "pkill -t %s" % re.split("/dev/", self._pty)[1]
            chan.exec_command(cmd)
            self._killed = True
            self._bad_stop(t, chan)
        except exceptions.Exception, e:
            msg.warning("can't kill tasks on %s : %s" % \
                            (self._host, e))

    def run(self):
        tm = self._tasks_manager
        hm = tm._hosts_manager
        task = self._task.get()

        # probe host
        hostname = self._host._name
        t, status = hm.probe(hostname)
        if t is None :
            msg.warning('probing %s failed' % hostname)
            #hm.update_host_status(hostname, status)
            self._bad_stop(t, None)
            return

        # get connection
        try:
            t.connect(**self._get_connection_params())
#FIXME : version 1.5.2 (at home) : n'existe pas, verifier a neurospin
#        except paramiko.AuthenticationException:
#            msg.warning('Authentication failed on %s' % hostname)
#            self._bad_stop(t, None)
#            return
        except paramiko.SSHException, e:
            msg.warning('ssh on %s: %s' % (hostname, e))
            self._bad_stop(t, None)
            return
        except exceptions.Exception, e:
            msg.warning('unknown exception (connection) : %s' % e)
            self._bad_stop(t, None)
            return

        # run task
        try:
            chan = t.open_session()
            if chan is None :
                msg.warning('ssh on %s' % hostname)
                self._bad_stop(t, None)
                return
        except paramiko.SSHException, e:
            msg.warning('ssh on %s: %s' % (hostname, e))
            self._bad_stop(t, None)
            return
        except exceptions.Exception, e:
            msg.warning('unknown exception (session opening) : %s' % e)
            self._bad_stop(t, None)
            return
        # associate a tty (pseudo terminal) to connection
        try:
            chan.get_pty()
        except paramiko.SSHException, e:
            msg.warning("can't get tty on %s : %s" % (hostname, e))
            self._bad_stop(t, None)
            return
        #time_limit = "86400" # 24h #FIXME : configurer ce truc
        mem_cmd_bash = "mem=$(cat /proc/meminfo | grep MemTotal " + \
                    "| sed \"s/.*: *//g;s/ kB//g\")"
        mem_cmd_csh = "setenv mem `cat /proc/meminfo | grep MemTotal "+\
                    "| sed \"s/.*: *//g;s/ kB//g\"`"
        limit_cmd_bash = "ulimit -v $mem -t %d" % self._time_limit
        limit_cmd_csh = "limit memoryuse $mem; " + \
                    "limit cputime %d" % self._time_limit
        pre_cmd_bash = "echo BASH;tty;%s;%s;if [ -a ~/.bashrc ]; then source ~/.bashrc;fi;" % \
                    (mem_cmd_bash, limit_cmd_bash)
        pre_cmd_csh = "echo CSH;tty;%s;%s;if [ -a ~/.cshrc ]; then source ~/.cshrc;fi;" % \
                    (mem_cmd_csh, limit_cmd_csh)
        pre_cmd = "(test `basename $SHELL` = 'bash' -o " + \
          "`basename $SHELL` = 'zsh') && eval '%s' || eval '%s';" % \
          (pre_cmd_bash, pre_cmd_csh)
          #print 'pre_cmd:', pre_cmd
        self._start_time = time.time()
        try:
            chan.exec_command(pre_cmd + task)
        except paramiko.SSHException, e:
            msg.warning('ssh on %s: %s' % (hostname, e))
            self._bad_stop(t, chan)
            return
        except exceptions.Exception, e:
            msg.warning('unknown exception : %s' % e)
            self._bad_stop(t, None)
            return

        self._read_pty(chan)
        self._status = chan.recv_exit_status()
        if self._status == -1 or self._status >= 128:
            self._remote_killed_stop(t, chan)
            return
        self._good_stop(t, chan)


class TasksManager(object):
    def __init__(self, timeslot, user, tasks, hosts_manager,
                     log, brokenfd, time_limit=86400):
        self._timeslot = timeslot
        self._user = user
        self._tasks = tasks
        self._hosts_manager = hosts_manager
        self._log = log
        self._brokenfd = brokenfd
        self._free_hosts = list(self._hosts_manager._available_list)
        self._active_hosts = []
        self._time_limit = time_limit

    def print_status(self, n, size):
        msg.write_list(['\r ', ('%d' % n, 'red'),
                '/', ('%d' % size, 'red')])
        if n == size: msg.write('\n')
        sys.stdout.flush()

    def wait_to_be_ready(self):
        we_have_waited = False
        if not self._timeslot.is_inside_now():
            we_have_waited = True
            now = time.asctime(time.localtime())
            msg.info('all tasks stopped : %s' % now)
        while not self._timeslot.is_inside_now():
            time.sleep(60)
        if we_have_waited:
            now = time.asctime(time.localtime())
            msg.info('start all remaining tasks : %s' % now)

    def abnormal_stop(self, task):
        self._tasks.append(task)


class DispatchedTasksManager(TasksManager):
    def __init__(self, *args, **kwargs):
        TasksManager.__init__(self, *args, **kwargs)
        self._tasks_finished_number = 0

    def wait_for_end_or_cmd(self, print_number, tasks_number, cmd):
        while 1:
            try: return cmd()
            except exceptions.IndexError: pass
            if self._tasks_finished_number == tasks_number:
                return None
            if len(self._free_hosts) == 0 and \
                len(self._active_hosts) == 0: return None
            if print_number[0] != self._tasks_finished_number:
                self.print_status(self._tasks_finished_number,
                    tasks_number)
                print_number[0] = self._tasks_finished_number
            time.sleep(0.01)

    def start(self):
        n = len(self._tasks)
        p = 0 # printed tasks number
        host_pop = self._free_hosts.pop
        while self._tasks_finished_number != n:
            host = self.wait_for_end_or_cmd([p], n, host_pop)
            get_next_task = self._tasks.next
            task = self.wait_for_end_or_cmd([p], n, get_next_task)
            if host and task:
                self.wait_to_be_ready()
                ts = TasksStarter(self, host, task,
                                                  self._time_limit)
                ts.start()
            if len(self._free_hosts) == 0 and \
                len(self._active_hosts) == 0: break
            if p != self._tasks_finished_number:
                self.print_status(self._tasks_finished_number,n)
                p = self._tasks_finished_number
            time.sleep(0.01)
        if len(self._free_hosts) == 0 and \
            self._tasks_finished_number != n :
            msg.warning('no more available hosts : sorry :(')


class RepeatedTasksManager(TasksManager):
    def __init__(self, *args, **kwargs):
        TasksManager.__init__(self, *args, **kwargs)
        self._tasks_finished_number = 0

    # FIXME : saute les machines temporairement en rade
    # FIXME : Si une machine est en rade, la tache est ajoutee :
    # self._tasks_manager._tasks.append(task)
    # alors que dans notre cas nous n'avons qu'une seule tache ou le
    # append est imposible. Ce sont les machines qu'il faudrait ajouter
    # a une liste de machine a retester eventuellement.
    def start(self):
        if len(self._tasks) != 1: raise exceptions.RuntimeError
        hosts = self._hosts_manager._available_list
        for host in hosts:
            self.wait_to_be_ready()
            ts = TasksStarter(self, host, self._tasks, self._time_limit)
            ts.start()
        while self._tasks_finished_number != len(hosts):
            self.print_status(self._tasks_finished_number,
                    len(hosts))
            time.sleep(0.01)
        self.print_status(self._tasks_finished_number, len(hosts))

    def abnormal_stop(self, task):
        self._tasks_finished_number += 1

class OneTaskManager(TasksManager):
    def __init__(self, *args, **kwargs):
        TasksManager.__init__(self, *args, **kwargs)
        self._tasks_finished_number = 0

    def start(self):
        if len(self._tasks) != 1: raise exceptions.RuntimeError
        hosts = self._hosts_manager._available_list
        if len(hosts) != 1: raise exceptions.RuntimeError
        self.wait_to_be_ready()
        ts = TasksStarter(self, hosts[0], self._tasks)
        ts.start()
        while len(self._task()) != 0: time.sleep(0.01)
        self.print_status(self._tasks_finished_number, 1)

    def abnormal_stop(self, task):
        pass # if task abnormally failed do nothing

class HierarchicalTasksManager(DispatchedTasksManager):
    def start(self):
        self._tasks.init()
        while 1:
            tasks = self._tasks.next()
            if tasks is None: break
            if isinstance(tasks, TaskHierarchical):
                rule = tasks.rule
                msg.write_list(['--> ', ('file', 'green'),
                    (" : '%s'\n" % rule)])
                htm = HierarchicalTasksManager(self._timeslot,
                    self._user, tasks, self._hosts_manager,
                    self._log, self._brokenfd)
                htm.start()
                msg.write_list(['<-- ', ('file', 'green'),
                    (" : '%s'\n" % rule)])
            else:
                dtm = DispatchedTasksManager(self._timeslot,
                    self._user, tasks, self._hosts_manager,
                    self._log, self._brokenfd)
                dtm.start()

def run_grid(mode, hosts_list, keytype, tasks, timeslot, brokenfile=None,
             logfile=None, user=None, passwd=None, time_limit=86400):
    if 0:
        print 'run_grid ...'
        print 'mode:', mode
        print 'hosts_list:', hosts_list
        print 'keytype:', keytype
        print 'tasks:', tasks
        print 'timeslot:', timeslot
        print 'brokenfile:', brokenfile
        print 'logfile:', logfile
        print 'user:', user
        print 'passwd:', passwd
        print 'time_limit:', time_limit
        print ''
        print ''
    hm = HostsManager(hosts_list)
    if len(hm._available_list) == 0:
        msg.error("All hosts down :(")
        return
    if tasks is None : return
        #print 'run_grid ...'
        #print 'tasks:', tasks
    brokenfd = open(brokenfile or os.devnull, 'w')
    log = open(logfile or os.devnull, 'w')
    user = User(user, passwd, keytype)
    args = (timeslot, user, tasks, hm, log, brokenfd, time_limit)
    if mode == 'dispatch': tm = DispatchedTasksManager(*args)
    elif mode == 'repeat': tm = RepeatedTasksManager(*args)
    elif mode in ['hie', 'hierarchic']:
        tm = HierarchicalTasksManager(*args)
    tm.start()
    brokenfd.close()
    log.close()

def main():
    # options
    parser = create_options(sys.argv)
    options, timeslot, hosts_list, tasks = parse_options(parser)
    run_grid(options.mode, hosts_list, options.keytype, tasks, timeslot,
             options.brokenfile, options.logfile, options.user,
             options.passwd)
    # hm = HostsManager(hosts_list)
    # if len(hm._available_list) == 0:
    #     msg.error("All hosts down :(")
    #     return
    # if tasks is None : return

    # brokenfd = open(options.brokenfile, 'w')
    # log = open(options.logfile, 'w')
    # user = User(options.user, options.passwd, options.keytype)
    # args = (timeslot, user, tasks, hm, log, brokenfd)
    # if options.mode == 'dispatch': tm = DispatchedTasksManager(*args)
    # elif options.mode == 'repeat': tm = RepeatedTasksManager(*args)
    # elif options.mode in ['hie', 'hierarchic']:
    #     tm = HierarchicalTasksManager(*args)
    # tm.start()
    # brokenfd.close()
    # log.close()

def kill_threads():
    print "\n"
    msg.info('kill threads')
    for t in threading.enumerate():
        if isinstance(t, TasksStarter):
            msg.info("kill task on '%s'" % t._host)
            t._write_broken()
            t.kill()
        t._Thread__stop()

def quit(signal, frame):
    kill_threads()
    sys.exit(1)


def main_safe():
    for sig in [signal.SIGQUIT, signal.SIGINT, signal.SIGSEGV]:
        signal.signal(sig, quit)
    try:
        main()
    except KeyboardInterrupt:
        quit(None, None)

if __name__ == "__main__" : main_safe()
