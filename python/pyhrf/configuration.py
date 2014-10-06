# -*- coding: utf-8 -*-
from __future__ import with_statement
import os.path as op
import os
from ConfigParser import RawConfigParser, ConfigParser

class ConfigurationError(Exception):
    pass

def write_configuration(cfgDict, fn, section_order=None):
    #print section_order
    if section_order is None:
        section_order = cfgDict.keys()
    cfgParser = RawConfigParser()
    for section in section_order:
        cfgParser.add_section(section)
        for key,value in cfgDict[section].iteritems():
            #print 'value:', value
            #print 'str(value):', str(value)
            cfgParser.set(section, key, str(value))

    with open(fn, 'wb') as configfile:
        cfgParser.write(configfile)

def cfg_error_report(cfg, refcfg):
    # Check if all names (section, pname) in cfg are in refcfg
    err = ''
    sections_to_pop = []
    for section in cfg.iterkeys():
        if section not in refcfg.keys():
            sections_to_pop.append(section)
            err += 'Undefined section: "%s"\n' %section
        else:
            labels_to_pop = []
            for pname in cfg[section].iterkeys():
                if pname not in refcfg[section].keys():
                    err += 'Undefined label "%s" (section "%s")\n' \
                        %(pname,section)
                    labels_to_pop.append(pname)
            for l in labels_to_pop:
                cfg[section].pop(l)
    for section in sections_to_pop:
        cfg.pop(section)

    return err

def load_configuration(fn, refcfg, mode='file_only'):
    """
    Load configuration file from 'fn' and check it against 'refcfg'.
    If mode is 'file_only' then only configuration in fn is returned.
    If mode is 'update' then the loaded configuration is updated with 'refcfg' to
    load defaults for unprovided parameters.
    """
    cfgParser = ConfigParser()
    cfgParser.read(fn)
    # Everything is stringified with ConfigParser
    # So cast everything to proper types: 
    cfg = {}
    for section in cfgParser.sections():
        cfg[section] = {}
        for pname, pval in cfgParser.items(section):
            cfg[section][pname] = pval
    err = cfg_error_report(cfg, refcfg)
    if err != '':
        msg = 'Errors in configuration file %s\n' %fn
        msg += err
        raise ConfigurationError(msg)

    for section in cfgParser.sections():
        for pname, pval in cfgParser.items(section):
            if isinstance(refcfg[section][pname],float):
                cfg[section][pname] = float(pval)
            elif isinstance(refcfg[section][pname],bool):
                cfg[section][pname] = (pval == 'True')
            elif isinstance(refcfg[section][pname],int):
                cfg[section][pname] = int(pval)
            if pval == 'None':
                cfg[section][pname] = None

    if mode == 'update':
        newcfg = refcfg.copy()
        newcfg.update(cfg)
        return newcfg
        
    return cfg

pyhrf_cfg_path = op.join(os.getenv('HOME'),'.pyhrf/')
if not op.exists(pyhrf_cfg_path):
    os.makedirs(pyhrf_cfg_path)


# Definition of default parameters
DEVEL = 0
ENDUSER = 1
useModes = {
    0 : 'devel',
    1 : 'end_user',
    }
useModesStr = {
    'devel' : 0,
    'end_user' : 1,
    }


defaults = { 
    'global' : {
        'verbosity' : 0,
        'use_mode' : 'end_user',
        'tmp_prefix' : 'pyhrftmp',
        'tmp_path' : '/tmp/',
        'spm_path' : None,
        'write_texture_minf' : False,
        'numpy_floating_point_error' : 'warn',
        },

    'parallel-LAN' : {
        'user' : None,
        'remote_host' : None,
        'hosts' : op.join(pyhrf_cfg_path,'hosts_LAN'),
        'remote_path' : None,
        'niceness' : 10,
        'enable_unit_test': 0,
        },
    'parallel-local' : {
        'niceness' : 10,
        'nb_procs' : 1,
        },
    'parallel-cluster' : {
        'server' : None,
        'server_id' : None,
        'user' : None,
        'remote_path' : None,
        'enable_unit_test': 0,
        },
    'treatment-default': {
        'save_result_dump' : 0,
        }
    }
section_order = ['global', 'parallel-LAN', 'parallel-local', 'parallel-cluster',
                 'treatment-default']

cfg = defaults.copy()

## Manage user-defined parameters
## in ~/.pyhrf/config.cfg

user_cfg_fn = op.join(pyhrf_cfg_path,'config.cfg')
if not op.exists(user_cfg_fn):
    write_configuration(defaults, user_cfg_fn, section_order)
else:
    try:
        cfg = load_configuration(user_cfg_fn, defaults, 'update')
    except ConfigurationError, e:
        print e
    except IOError, e:
        print "Errors reading configuration file:"
        print e
    for section in defaults.iterkeys():
        default_keys = defaults[section].keys()
        unknownLabels = set(default_keys).difference(cfg[section].keys())
        if len(unknownLabels) > 0:
            for l in unknownLabels:
                cfg[section][l] = defaults[section][l]
            write_configuration(cfg, user_cfg_fn, section_order)
