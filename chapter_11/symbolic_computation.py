import sympy as sy


########################
# integration

x = sy.Symbol('x')
y = sy.Symbol('y')

print(sy.sqrt(x))   # note how despite x having no numerical value, this still works
print(3 + sy.sqrt(x) - 4 ** 2)
 
f = x ** 2 + 3 + -.5 * x ** 2 + 3/2
print (sy.simplify(f))

print(sy.pretty(f))
print(sy.pretty(sy.sqrt(x) + 0.5))


# equations work too
print(sy.solve(x ** 2 - 1))         # real solution
print(sy.solve(x ** 2 + y ** 2))    # imaginary solution

# integration and differentiation
a, b = sy.symbols('a b')    # symbols for integral limits
I = sy.Integral(
    sy.sin(x) + 0.5 * x,    # actual Integral
    (x, a, b)
)
print(sy.pretty(I))

int_func = sy.integrate(sy.sin(x) + 0.5 * x, x) # antiderivative     
print(sy.pretty(int_func))

Fb = int_func.subs(x, 9.5).evalf()  # antiderivate at interval limits
Fa = int_func.subs(x, 0.5).evalf()
print(Fb - Fa)  # numeric value of integral

# or in one step
print(sy.integrate(
    sy.sin(x) + 0.5 * x,    # integral func 
    (x, 0.5, 9.5)           # interval limits + variable
    )
)


########################
# differentiation
print(int_func.diff())  # back to where we started

# trying to find a global min - see convex minimization problem
f = (sy.sin(x) + 0.05 * x ** 2 
     + sy.sin(y) + 0.05 * y ** 2)

del_x = sy.diff(f, x)
print(del_x)
del_y = sy.diff(f, y)
print(del_y)

x0 = sy.nsolve(del_x, -1.5) # educated guess for the roots 
y0 = sy.nsolve(del_y, -1.5) # not having an educated guess might trap the algo in a local min
                            # try the same w 1.5 instead of negative
print(f.subs({x: x0, y: y0}).evalf())   # global min function value

