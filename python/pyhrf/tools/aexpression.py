# -*- coding: utf-8 -*-


from math import *
#import collections
from pyhrf.tools import apply_to_leaves
import re
from copy import deepcopy
"""
Mathematical Expression Evaluator class for Python.
You can set the expression member, set the functions, variables and then call
evaluate() function that will return you the result of the mathematical expression
given as a string. 
 
The user is granted rights to user, distribute and/or modify the source code 
as long as this notice is shipped with it.
 
The Author of this software cannot and do not warrant that any 
functions contained in the Software will meet your requirements, 
or that its operations will be error free. 
 
The entire risk as to the Software performance or quality, 
or both, is solely with the user and not the Author.
 
You assume responsibility for the selection of the software to 
achieve your intended results, and for the installation, use, and 
results obtained from the Software. 
 
The Author makes no warranty, either implied or expressed, including 
with-out limitation any warranty with respect to this Software 
documented here, its quality, performance, or fitness for a particular 
purpose. In no event shall the Author be liable to you for damages, 
whether direct or indirect, incidental, special, or consequential 
arising out the use of or any defect in the Software, even if the 
Author has been advised of the possibility of such damages, 
or for any claim by any other party. 
 
All other warranties of any kind, either express or implied, 
including but not limited to the implied warranties of 
merchantability and fitness for a particular purpose, are expressly 
excluded.
 
Copyright and all rights of this software, irrespective of what
has been deposited with the U.S. Copyright Office, belongs
to Bestcode.com.
"""

class ArithmeticExpressionSyntaxError(Exception):
    pass

class ArithmeticExpressionNameError(Exception):
    pass


class ArithmeticExpression(object):
    '''
    Mathematical Expression Evaluator class.
    You can set the expression member, set the functions, variables and then call
    evaluate() function that will return you the result of the mathematical 
    expression given as a string.
    '''
   
    '''
    Dictionary of functions that can be used in the expression.
    '''
    functions = None #{'__builtins__':None};
   
    '''
    Dictionary of variables that can be used in the expression.
    '''
    variables = None #{'__builtins__':None};

    def __init__(self, expression, **variables):
        ''' Constructor '''
        self.expression = expression

        if self.variables is None:
            self.variables = {'__builtins__':None}

        if self.functions is None:
            functions = {'__builtins__':None};

        for vname, val in variables.iteritems():
            self.variables[vname] = val
   
    def call_if_func(self, x):
        if hasattr(x, '__call__'): # or isinstance(x, collections.Callable):
            return x()
        else: return x

    def evaluate(self):
        '''
        Evaluate the mathematical expression given as a string in the 
        expression member variable.
       
        '''
        variables = apply_to_leaves(self.variables, self.call_if_func)
        try:
            result = eval(self.expression, variables, self.functions)
        except NameError, err:
            undef_var = re.findall("name '(.*)' is not defined", err.args[0])[0]
            raise NameError("Variable '%s' is not defined in expression '%s'" \
                                %(undef_var, self.expression), undef_var)
        
        return result


    def check(self):
        
        #print 'self.variables:', self.variables
        try:
            self.evaluate()
        except Exception ,err :
            m = 'Error in expression "%s": ' %self.expression
            if isinstance(err, NameError):
                exception_occurs = True
                undefs = []
                old_vars = self.variables.copy()
                while exception_occurs:
                    exception_occurs = False
                    try:
                        self.evaluate()
                    except NameError, err :
                        exception_occurs = True
                        v = self.variables
                        undefs.append(err.args[1])
                        if len(v) == 1: # only default __builtins__
                            dummy_val = 1
                        elif v.keys()[0] == '__builtins__':
                            dummy_val = v[v.keys()[1]]
                        else:
                            dummy_val = v[v.keys()[0]]
                        v[err.args[1]] = dummy_val
                m += ' *undefined variable(s): %s' %','.join(undefs)
                self.variables = old_vars
                raise ArithmeticExpressionNameError(m, self.expression,
                                                    undefs)
            elif isinstance(err, SyntaxError):
                m += ' *syntax error at offset %d' %err.offset
                raise ArithmeticExpressionSyntaxError(m, self.expression,
                                                      err.offset)
            else:
                raise err

    def addDefaultFunctions(self):
        '''
        Add the following Python functions to be used in a mathemtical expression:
        acos
        asin
        atan
        atan2
        ceil
        cos
        cosh
        degrees
        exp
        fabs
        floor
        fmod
        frexp
        hypot
        ldexp
        log
        log10
        modf
        pow
        radians
        sin
        sinh
        sqrt
        tan
        tanh
        '''
        self.functions['acos']=acos
        self.functions['asin']=asin
        self.functions['atan']=atan
        self.functions['atan2']=atan2
        self.functions['ceil']=ceil
        self.functions['cos']=cos
        self.functions['cosh']=cosh
        self.functions['degrees']=degrees
        self.functions['exp']=exp
        self.functions['fabs']=fabs
        self.functions['floor']=floor
        self.functions['fmod']=fmod
        self.functions['frexp']=frexp
        self.functions['hypot']=hypot
        self.functions['ldexp']=ldexp
        self.functions['log']=log
        self.functions['log10']=log10
        self.functions['modf']=modf
        self.functions['pow']=pow
        self.functions['radians']=radians
        self.functions['sin']=sin
        self.functions['sinh']=sinh
        self.functions['sqrt']=sqrt
        self.functions['tan']=tan
        self.functions['tanh']=tanh
       
    def addDefaultVariables(self):
        '''
        Add e and pi to the list of defined variables.
        '''
        self.variables['e']=e
        self.variables['pi']=pi

    def setVariable(self, name, value):
        '''
        Define the value of a variable defined by name
        '''
        self.variables[name] = value

    def getVariableNames(self):
        '''
        Return a List of defined variables names in sorted order.
        '''
        mylist = list(self.variables.keys())
        try:
            mylist.remove('__builtins__')
        except ValueError:
            pass
        mylist.sort()
        return mylist


    def getFunctionNames(self):
        '''
        Return a List of defined function names in sorted order.
        '''
        mylist = list(self.functions.keys())
        try:
            mylist.remove('__builtins__')
        except ValueError:
            pass
        mylist.sort()
        return mylist
