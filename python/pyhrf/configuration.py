# -*- coding: utf-8 -*-

"""Loads and allows configuration of PyHRF."""

from __future__ import with_statement

import os
import os.path as op

from ConfigParser import RawConfigParser, ConfigParser


pyhrf_cfg_path = op.join(os.getenv('HOME'), '.pyhrf/')
if not op.exists(pyhrf_cfg_path):
    os.makedirs(pyhrf_cfg_path)

# Definition of default parameters
DEVEL = 0
ENDUSER = 1
useModesStr = {
    'devel': DEVEL,
    'end_user': ENDUSER,
}
useModes = {useModesStr[k]: k for k in useModesStr}

defaults = {
    'global': {
        'verbosity': 0,
        'use_mode': 'end_user',
        'tmp_prefix': 'pyhrftmp',
        'tmp_path': '/tmp/',
        'spm_path': None,
        'write_texture_minf': False,
        'numpy_floating_point_error': 'warn',
    },
    'log': {
        'level': 'WARNING',
        'log_to_file': False,
    },

    'parallel-LAN': {
        'user': None,
        'remote_host': None,
        'hosts': op.join(pyhrf_cfg_path, 'hosts_LAN'),
        'remote_path': None,
        'niceness': 10,
        'enable_unit_test': 0,
    },
    'parallel-local': {
        'niceness': 10,
        'nb_procs': 1,
    },
    'parallel-cluster': {
        'server': None,
        'server_id': None,
        'user': None,
        'remote_path': None,
        'enable_unit_test': 0,
    },
    'treatment-default': {
        'save_result_dump': 0,
    }
}

section_order = ['global',
                 'log',
                 'parallel-LAN',
                 'parallel-local',
                 'parallel-cluster',
                 'treatment-default']

cfg = defaults.copy()


class ConfigurationError(Exception):
    """Exception class for configuration parsing errors."""
    pass


def write_configuration(cfg_dict, filename, section_order=None):
    if section_order is None:
        section_order = cfg_dict.keys()
    cfgParser = RawConfigParser()
    for section in section_order:
        cfgParser.add_section(section)
        for key, value in cfg_dict[section].iteritems():
            # print 'value:', value
            # print 'str(value):', str(value)
            cfgParser.set(section, key, str(value))

    with open(filename, 'wb') as configfile:
        cfgParser.write(configfile)


def cfg_error_report(cfg, refcfg):
    # Check if all names (section, pname) in cfg are in refcfg
    err = ''
    sections_to_pop = []
    for section in cfg.iterkeys():
        if section not in refcfg.keys():
            sections_to_pop.append(section)
            err += 'Undefined section: "%s"\n' % section
        else:
            labels_to_pop = []
            for pname in cfg[section].iterkeys():
                if pname not in refcfg[section].keys():
                    err += 'Undefined label "%s" (section "%s")\n' \
                        % (pname, section)
                    labels_to_pop.append(pname)
            for l in labels_to_pop:
                cfg[section].pop(l)
    for section in sections_to_pop:
        cfg.pop(section)

    return err


def load_configuration(filename, refcfg, mode='file_only'):
    """
    Load configuration file from 'filename' and check it against 'refcfg'.
    If mode is 'file_only' then only configuration in filename is returned.
    If mode is 'update' then the loaded configuration is updated with 'refcfg' to
    load defaults for unprovided parameters.
    """
    cfgParser = ConfigParser()
    cfgParser.read(filename)
    # Everything is stringified with ConfigParser
    # So cast everything to proper types:
    cfg = {}
    for section in cfgParser.sections():
        cfg[section] = {}
        for pname, pval in cfgParser.items(section):
            cfg[section][pname] = pval
    err = cfg_error_report(cfg, refcfg)
    if err != '':
        msg = 'Errors in configuration file %s\n' % filename
        msg += err
        raise ConfigurationError(msg)

    for section in cfgParser.sections():
        for pname, pval in cfgParser.items(section):
            if isinstance(refcfg[section][pname], float):
                cfg[section][pname] = float(pval)
            elif isinstance(refcfg[section][pname], bool):
                cfg[section][pname] = (pval == 'True')
            elif isinstance(refcfg[section][pname], int):
                cfg[section][pname] = int(pval)
            if pval == 'None':
                cfg[section][pname] = None

    if mode == 'update':
        newcfg = refcfg.copy()
        newcfg.update(cfg)
        return newcfg

    return cfg


# Manage user-defined parameters
# in ~/.pyhrf/config.cfg

user_cfg_fn = op.join(pyhrf_cfg_path, 'config.cfg')
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
