import turtle
from svg_turtle import *

import os, sys
import re, glob
import numpy as np


EPS = 1e-3
FACTOR = 5
WIDTH, HEIGHT = 500, 500

DEBUG = True
ARGS = ['draw', 'svg']


try:
    assert sys.argv[1] in ARGS
except:
    print('Use one of the following arguments: {}!'.format(', '.join(ARGS)))
    sys.exit()


if sys.argv[1] == 'draw':
    s = turtle.Screen()
    s.setup(2 * (WIDTH * FACTOR), 2 * (HEIGHT * FACTOR))
    s.screensize(2 * (WIDTH * FACTOR), 2 * (HEIGHT * FACTOR))
    s.setworldcoordinates(-WIDTH * FACTOR, -HEIGHT * FACTOR, +WIDTH * FACTOR, +HEIGHT * FACTOR)

    t = turtle.Turtle()
elif sys.argv[1] == 'svg':
    t = SvgTurtle(2 * (WIDTH * FACTOR) + 1, 2 * (HEIGHT * FACTOR) + 1)


def scale(*args):
    return np.multiply(args, FACTOR)

def animate(flag=None):
    if flag is None: s.tracer()
    else: s.tracer(flag, 10)

def reset(x, y, o):
    t.penup()
    t.home()
    t.setposition(x, y)
    if o == 'x': t.left(0)
    if o == 'y': t.left(90)
    t.pendown()

def check(x, y, o, l, c):
    if o == 'x':
        i = 0
        v = x + l
    if o == 'y':
        i = 1
        v = y + l

    if c == 'gt': return t.position()[i] - EPS >= v
    if c == 'lt': return t.position()[i] + EPS <= v

def grid(e):
    e, = scale(e)

    t.pencolor('yellow')
    for x in range(-WIDTH, +WIDTH + 1, e):
        straight(x, -HEIGHT, 2 * HEIGHT, 'y')
    for y in range(-HEIGHT, +HEIGHT + 1, e):
        straight(-WIDTH, y, 2 * WIDTH, 'x')
    t.pencolor('black')

    straight(-e, 0, 2 * e, 'x')
    straight(0, -e, 2 * e, 'y')

# t.clone()
# t.heading()
# t.clear()
# t.clearstamp()
# t.degrees()
# t.distance()
# t.done()
# t.goto()
# t.hideturtle()
# t.pencolor()
# t.pensize()
# t.position()
# t.radians()
# t.reset()
# t.setheading()
# t.setposition()
# t.shape()
# t.shapesize()
# t.shapetransform()
# t.speed()
# t.stamp()
# t.towards()











































def straight(x, y, o, l):
    x, y, l = scale(x, y, l)

    reset(x, y, o)
    t.forward(l)

def circle(x, y, o, r, s=None):
    x, y, r = scale(x, y, r)

    reset(x, y, o)
    t.circle(r, steps=s)

def ellipsis(x, y, o, rx, ry, s=None):
    x, y, rx, ry = scale(x, y, rx, ry)

    reset(x, y, o)
    if o == 'x':
        t.right(45)
        for _ in range(2):
            t.circle(rx, 90, steps=s)
            t.circle(ry, 90, steps=s)
    if o == 'y':
        t.right(45)
        for _ in range(2):
            t.circle(ry, 90, steps=s)
            t.circle(rx, 90, steps=s)

def polygon(x, y, o, k, d):
    x, y, k = scale(x, y, k)

    reset(x, y, o)
    for _ in range(d):
        t.forward(k)
        t.left(360/d)

# == skyline
def flower(x, y, o, n, r, a, s=None):
    x, y, r = scale(x, y, r)

    reset(x, y, o)
    for _ in range(n):
        t.circle(r, a, steps=s)
        t.left(360/n-a)

def flower_poly(x, y, o, n, k, a, d):
    x, y, k = scale(x, y, k)

    reset(x, y, o)
    for _ in range(n):
        t.circle(r, 180, steps=s)
        t.right(170)

# == slalom
def gear(x, y, o, n, r1, r2, a1, a2, s=None):
    x, y, r = scale(x, y, r)

    reset(x, y, o)
    for _ in range(n):
        t.circle(+r1, a1, steps=s)
        t.circle(-r2, a2, steps=s)

def gear_poly(x, y, o, n, k1, k2, a1, a2, d):
    x, y, k = scale(x, y, k)

    reset(x, y, o)
    for _ in range(n):
        t.forward(k-k/d)
        for _ in range(d):
            t.left(180/d)
            t.forward(k/d)

# == signal
def rounded(x, y, o, n, r, s=None):
    x, y, r = scale(x, y, r)

    reset(x, y, o)
    for _ in range(n):
        t.forward(r*np.pi/n)
        t.circle(r, 360/n, steps=s)

def rounded_poly(x, y, o, n, k, a, d):
    x, y, k = scale(x, y, k)

    reset(x, y, o)
    for _ in range(n):
        t.forward(k-k/d)
        for _ in range(d):
            t.left(180/d)
            t.forward(k/d)

# == up_n_down
def puzzle(x, y, o, n, k, r, a, s=None):
    x, y, k, r = scale(x, y, k, r)

    reset(x, y, o)
    for _ in range(n):
        t.forward(r*np.pi/n)
        t.circle(r, 360/n, steps=s)

def puzzle_poly(x, y, o, n, k, a, d):
    x, y, k = scale(x, y, k)

    reset(x, y, o)
    for _ in range(n):
        t.forward(k-k/d)
        for _ in range(d):
            t.left(180/d)
            t.forward(k/d)

def skyline(x, y, o, l, r, a, s=None):
    x, y, l, r = scale(x, y, l, r)

    reset(x, y, o)
    if o == 'x': t.left(a/2)
    if o == 'y': t.right(a/2)
    while True:
        if check(x, y, o, l, 'gt'): break
        t.circle(r, a, steps=s)

def skyline_poly(x, y, o, l, k, a, d):
    x, y, l, k = scale(x, y, l, k)

    reset(x, y, o)
    if o == 'x': pass
    if o == 'y': t.right(360/d)
    while True:
        if check(x, y, o, l, 'gt'): break
        for i in range(d):
            t.forward(k)
            if i == d - 2: break
            t.left(360/d)

def slalom(x, y, o, l, r, a, s=None):
    x, y, l, r = scale(x, y, l, r)

    reset(x, y, o)
    if o == 'x': t.left(a/2)
    if o == 'y': t.right(a/2)
    while True:
        if check(x, y, o, l, 'gt'): break
        t.circle(+r, a, steps=s)
        if check(x, y, o, l, 'gt'): break
        t.circle(-r, a, steps=s)

def slalom_poly(x, y, o, l, k, a, d):
    x, y, l, k = scale(x, y, l, k)

    reset(x, y, o)
    if o == 'x': pass
    if o == 'y': t.right(360/d)
    while True:
        if check(x, y, o, l, 'gt'): break
        for i in range(d):
            t.forward(k)
            if i == d - 2: break
            t.left(360/d)
        if check(x, y, o, l, 'gt'): break
        for i in range(d):
            t.forward(k)
            if i == d - 2: break
            t.right(360/d)

def signal(x, y, o, l, r, a, s=None):
    x, y, l, r = scale(x, y, l, r)

    a = 200
    reset(x, y, o)
    while True:
        if check(x, y, o, l, 'gt'): break
        t.forward(r*np.pi/(180/a))
        t.right(a/2)
        if check(x, y, o, l, 'gt'): break
        t.circle(r, a, steps=s)
        t.right(a/2)

def signal_poly(x, y, o, l, k, a, d):
    x, y, l, r = scale(x, y, l, r)

    reset(x, y, o)
    while True:
        if check(x, y, o, l, 'gt'): break
        t.forward(r*np.pi/(180/a))
        t.right(a)
        if check(x, y, o, l, 'gt'): break
        t.circle(r, a, steps=s)
        t.right(a)

def up_n_down(x, y, o, l, r, a, s=None):
    x, y, l, r = scale(x, y, l, r)

    reset(x, y, o)
    while True:
        if check(x, y, o, l, 'gt'): break
        t.forward(r*np.pi/(180/a))
        t.right(a)
        if check(x, y, o, l, 'gt'): break
        t.circle(+r, a, steps=s)
        t.right(a)
        if check(x, y, o, l, 'gt'): break
        t.forward(r*np.pi/(180/a))
        t.left(a)
        if check(x, y, o, l, 'gt'): break
        t.circle(-r, a, steps=s)
        t.left(a)

def up_n_down_poly(x, y, o, l, k, a, d):
    x, y, l, r = scale(x, y, l, r)

    reset(x, y, o)
    while True:
        if check(x, y, o, l, 'gt'): break
        t.forward(r*np.pi/(180/a))
        t.right(a)
        if check(x, y, o, l, 'gt'): break
        t.circle(+r, a, steps=s)
        t.right(a)
        if check(x, y, o, l, 'gt'): break
        t.forward(r*np.pi/(180/a))
        t.left(a)
        if check(x, y, o, l, 'gt'): break
        t.circle(-r, a, steps=s)
        t.left(a)

def snake(x, y, o, l, r, a, s=None):
    x, y, l, r = scale(x, y, l, r)

    reset(x, y, o)
    if o == 'x': t.left(a/2)
    if o == 'y': t.right(a/2)
    while True:
        if check(x, y, o, l, 'gt'): break
        t.forward(r*np.pi/2)
        if check(x, y, o, l-5, 'gt'): break
        t.circle(+r, a, steps=s)
        if check(x, y, o, l, 'gt'): break
        t.forward(r*np.pi/2)
        if check(x, y, o, l-5, 'gt'): break
        t.circle(-r, a, steps=s)

def snake_poly(x, y, o, l, k, a, d):
    x, y, l, k = scale(x, y, l, k)

    reset(x, y, o)
    if o == 'x': t.left(a/2)
    if o == 'y': t.right(a/2)
    while True:
        if check(x, y, o, l, 'gt'): break
        t.forward(k)
        t.left(90)
        if check(x, y, o, l, 'gt'): break
        t.forward(k)
        t.left(90)
        if check(x, y, o, l, 'gt'): break
        t.forward(k)
        t.right(90)
        if check(x, y, o, l, 'gt'): break
        t.forward(k)
        t.right(90)

def wave_lr(x, y, o, l, r, a, s=None):
    x, y, l, r = scale(x, y, l, r)

    reset(x, y, o)
    if o == 'x': pass
    if o == 'y': pass
    t.left(52)
    # t.left(a/np.pi)
    while True:
        if check(x, y, o, l, 'gt'): break
        t.forward(r*np.pi/(180/a))
        if check(x, y, o, l, 'gt'): break
        t.circle(-r, a, steps=s)
        if check(x, y, o, l, 'gt'): break
        t.circle(+r, a, steps=s)

def wave_rl(x, y, o, l, r, a, s=None):
    x, y, l, r = scale(x, y, l, r)

    reset(x, y, o)
    if o == 'x': pass
    if o == 'y': pass
    t.right(52)
    # t.right(a/np.pi)
    while True:
        if check(x, y, o, l, 'gt'): break
        t.forward(r*np.pi/(180/a))
        if check(x, y, o, l, 'gt'): break
        t.circle(+r, a, steps=s)
        if check(x, y, o, l, 'gt'): break
        t.circle(-r, a, steps=s)

def wave_lr_poly(x, y, o, l, r, a, s=None):
    x, y, l, r = scale(x, y, l, r)

    reset(x, y, o)
    if o == 'x': pass
    if o == 'y': pass
    t.left(52)
    # t.left(a/np.pi)
    while True:
        if check(x, y, o, l, 'gt'): break
        t.forward(r*np.pi/(180/a))
        if check(x, y, o, l, 'gt'): break
        t.circle(-r, a, steps=s)
        if check(x, y, o, l, 'gt'): break
        t.circle(+r, a, steps=s)

def wave_rl_poly(x, y, o, l, r, a, s=None):
    x, y, l, r = scale(x, y, l, r)

    reset(x, y, o)
    if o == 'x': pass
    if o == 'y': pass
    t.right(52)
    # t.right(a/np.pi)
    while True:
        if check(x, y, o, l, 'gt'): break
        t.forward(r*np.pi/(180/a))
        if check(x, y, o, l, 'gt'): break
        t.circle(+r, a, steps=s)
        if check(x, y, o, l, 'gt'): break
        t.circle(-r, a, steps=s)

def mesh(x, y, o, l, r, a, s=None):
    x, y, l, r = scale(x, y, l, r)

    reset(x, y, o)
    while True:
        if check(x, y, o, l, 'gt'): break
        t.forward(r*np.pi)
        if check(x, y, o, l-5, 'gt'): break
        t.circle(+r, a, steps=s)
        if check(x, y, o, l-5, 'gt'): break
        t.circle(-r, a, steps=s)
        if check(x, y, o, l, 'gt'): break
        t.forward(r*np.pi)
        if check(x, y, o, l-5, 'gt'): break
        t.circle(-r, a, steps=s)
        if check(x, y, o, l-5, 'gt'): break
        t.circle(+r, a, steps=s)

def mesh_poly(x, y, o, l, k, a, d):
    x, y, l, k = scale(x, y, l, k)

    reset(x, y, o)
    while True:
        if check(x, y, o, l, 'gt'): break
        t.forward(r*np.pi)
        if check(x, y, o, l, 'gt'): break
        t.circle(-r, a, steps=s)
        if check(x, y, o, l, 'gt'): break
        t.circle(+r, a, steps=s)
        if check(x, y, o, l, 'gt'): break
        t.forward(r*np.pi)
        if check(x, y, o, l, 'gt'): break
        t.circle(+r, a, steps=s)
        if check(x, y, o, l, 'gt'): break
        t.circle(-r, a, steps=s)

def spiral_in_circle(x, y, o, f, r, s=None):
    x, y, f, r = scale(x, y, f, r)

    reset(x, y, o)

    # a = 0
    # while a < 360 * 5:
    #     r = a / (2 * np.pi)
    #     a += 1
    #     # t.circle(r / (1 + np.log(i)), 100 / (1 + np.log(i)))
    #     # t.circle(r / (1 + i / 1e1), 100 / (1 + i / 1e1))
    #     t.circle(r, np.radians(a))

    d = 0.5
    a = 0

    t.penup()
    t.goto(d * a / (2 * np.pi) * np.cos(np.radians(a)) + x, d * a / (2 * np.pi) * np.sin(np.radians(a)) + y)
    t.pendown()
    while a < 360 * 10:
        r = d * a / (2 * np.pi)
        t.goto(r * np.cos(np.radians(a)) + x, r * np.sin(np.radians(a)) + y)
        a += 1

    # a = 0
    # d = 25
    # while r > 0:
    #     t.circle(r, a)
    #     r -= d
    #     a += 10

def spiral_in_ellipsis(x, y, o, f, rx, ry, s=None):
    x, y, f, rx, ry = scale(x, y, f, rx, ry)

    reset(x, y, o)
    pass

def spiral_in_polygon(x, y, o, f, k, d):
    x, y, f, k = scale(x, y, f, k)

    reset(x, y, o)
    while True:
        for _ in range(d):
            t.forward(l)
            t.left(360/d)
            l -= s
            if l <= 0: return

def spiral_in_flower(x, y, o, f, n, r, a, s):
    x, y, f, r = scale(x, y, f, r)

    reset(x, y, o)
    for _ in range(d):
        t.forward(r*np.pi)
        t.circle(r, 360/d)

def spiral_in_flower_poly(x, y, o, f, n, k, a, d):
    x, y, f, k = scale(x, y, f, k)

    reset(x, y, o)
    for _ in range(d):
        t.forward(l)
        t.circle(l/np.pi, 360/d)

def spiral_in_gear(x, y, o, f, n, r, a, s):
    x, y, f, r = scale(x, y, f, r)

    reset(x, y, o)
    for _ in range(d):
        t.forward(r*np.pi)
        t.circle(r, 360/d)

def spiral_in_gear_poly(x, y, o, f, n, k, a, d):
    x, y, f, k = scale(x, y, f, k)

    reset(x, y, o)
    for _ in range(d):
        t.forward(l)
        t.circle(l/np.pi, 360/d)

def spiral_in_rounded(x, y, o, f, n, r, a, s):
    x, y, f, r = scale(x, y, f, r)

    reset(x, y, o)
    for _ in range(d):
        t.forward(r*np.pi)
        t.circle(r, 360/d)

def spiral_in_rounded_poly(x, y, o, f, n, k, a, d):
    x, y, f, k = scale(x, y, f, k)

    reset(x, y, o)
    for _ in range(d):
        t.forward(l)
        t.circle(l/np.pi, 360/d)

def spiral_in_puzzle(x, y, o, f, n, r, a, s):
    x, y, f, r = scale(x, y, f, r)

    reset(x, y, o)
    for _ in range(d):
        t.forward(r*np.pi)
        t.circle(r, 360/d)

def spiral_in_puzzle_poly(x, y, o, f, n, k, a, d):
    x, y, f, k = scale(x, y, f, k)

    reset(x, y, o)
    for _ in range(d):
        t.forward(l)
        t.circle(l/np.pi, 360/d)

def spiral_out_circle(x, y, o, f, r, s=None):
    x, y, f, r = scale(x, y, f, r)

    reset(x, y, o)

    # a = 0
    # while a < 360 * 5:
    #     r = a / (2 * np.pi)
    #     a += 1
    #     # t.circle(r / (1 + math.log(i)), 100 / (1 + math.log(i)))
    #     # t.circle(r / (1 + i / 1e1), 100 / (1 + i / 1e1))
    #     t.circle(r, np.radians(a))

    # d = 1
    # a = 0
    # while a < 360 * 10:
    #     r = d * a / (2 * math.pi)
    #     t.goto(r * math.cos(math.radians(a)) + x, r * math.sin(math.radians(a)) - y)
    #     a += 1

    # a = 0
    # d = 25
    # while r > 0:
    #     t.circle(r, a)
    #     r -= d
    #     a += 10

def spiral_out_ellipsis(x, y, o, f, rx, ry, s=None):
    x, y, f, rx, ry = scale(x, y, f, rx, ry)

    reset(x, y, o)
    pass

def spiral_out_polygon(x, y, o, f, k, d):
    x, y, f, k = scale(x, y, f, k)

    reset(x, y, o)
    while True:
        for _ in range(d):
            t.forward(l)
            t.left(360/d)
            l -= s
            if l <= 0: return

def spiral_out_flower(x, y, o, f, n, r, a, s):
    x, y, f, r = scale(x, y, f, r)

    reset(x, y, o)
    for _ in range(d):
        t.forward(r*np.pi)
        t.circle(r, 360/d)

def spiral_out_flower_poly(x, y, o, f, n, k, a, d):
    x, y, f, k = scale(x, y, f, k)

    reset(x, y, o)
    for _ in range(d):
        t.forward(l)
        t.circle(l/np.pi, 360/d)

def spiral_out_gear(x, y, o, f, n, r, a, s):
    x, y, f, r = scale(x, y, f, r)

    reset(x, y, o)
    for _ in range(d):
        t.forward(r*np.pi)
        t.circle(r, 360/d)

def spiral_out_gear_poly(x, y, o, f, n, k, a, d):
    x, y, f, k = scale(x, y, f, k)

    reset(x, y, o)
    for _ in range(d):
        t.forward(l)
        t.circle(l/np.pi, 360/d)

def spiral_out_rounded(x, y, o, f, n, r, a, s):
    x, y, f, r = scale(x, y, f, r)

    reset(x, y, o)
    for _ in range(d):
        t.forward(r*np.pi)
        t.circle(r, 360/d)

def spiral_out_rounded_poly(x, y, o, f, n, k, a, d):
    x, y, f, k = scale(x, y, f, k)

    reset(x, y, o)
    for _ in range(d):
        t.forward(l)
        t.circle(l/np.pi, 360/d)

def spiral_out_puzzle(x, y, o, f, n, r, a, s):
    x, y, f, r = scale(x, y, f, r)

    reset(x, y, o)
    for _ in range(d):
        t.forward(r*np.pi)
        t.circle(r, 360/d)

def spiral_out_puzzle_poly(x, y, o, f, n, k, a, d):
    x, y, f, k = scale(x, y, f, k)

    reset(x, y, o)
    for _ in range(d):
        t.forward(l)
        t.circle(l/np.pi, 360/d)





def slalom_special(x, y, o, l, r, a, s=None):
    x, y, l, r = scale(x, y, l, r)

    reset(x, y, o)
    if o == 'x': t.left(a/2)
    if o == 'y': t.right(a/2)
    while True:
        r -= 3.8
        if r < 0: break
        if check(x, y, o, l, 'gt'): break
        t.circle(+r, a, steps=s)
        if check(x, y, o, l, 'gt'): break
        t.circle(-r, a, steps=s)

def slalom_special_double(x, y, o, l, r, a, s=None):
    x, y, l, r = scale(x, y, l, r)

    org_r = r
    reset(x, y, o)
    if o == 'x': t.left(a/2)
    if o == 'y': t.right(a/2)
    flag = True
    while True:
        if flag: r -= 8.05
        if r < 8.05: flag = False
        if not flag: r += 8.05
        if r > org_r: break
        if check(x, y, o, l, 'gt'): break
        t.circle(+r, a, steps=s)
        if check(x, y, o, l, 'gt'): break
        t.circle(-r, a, steps=s)

def spiral_in_circle_special_r(x, y, o, f, r, s=None):
    x, y, f, r = scale(x, y, f, r)

    reset(x, y, o)
    d = 0.5
    a = 0

    sign = +1
    if r < 0: sign = -1

    t.penup()
    t.goto(d * a / (2 * np.pi) * np.cos(np.radians(a)) + x, d * a / (2 * np.pi) * np.sin(np.radians(a)) + y)
    t.pendown()
    while a < 360 * 10 + 1:
        r = d * a / (2 * np.pi)
        t.goto(r * np.cos(np.radians(sign*a)) + x, r * np.sin(np.radians(sign*a)) + y)
        a += 1

def spiral_in_circle_special_l(x, y, o, f, r, s=None):
    x, y, f, r = scale(x, y, f, r)

    reset(x, y, o)
    d = 0.5
    a = 0

    sign = +1
    if r < 0: sign = -1

    t.penup()
    t.goto(d * a / (2 * np.pi) * np.cos(np.radians(a)) + x, d * a / (2 * np.pi) * np.sin(np.radians(a)) + y)
    t.pendown()
    while a < 360 * 10 + 1:
        r = d * a / (2 * np.pi)
        t.goto(x + r * np.cos(np.radians(sign*(a + 180))), y + r * np.sin(np.radians(sign*(a + 180))))
        a += 1

def spiral_in_circle_special_double_r(x, y, o, f, r, s=None):
    x, y, f, r = scale(x, y, f, r)

    reset(x, y, o)
    # # Setting the angle between the spiral arms
    # phi = 137.508

    # # Loop to draw the spiral
    # for i in range(200):
    #     angle = i * phi
    #     x = (r * np.sqrt(i)) * np.cos(np.radians(angle))
    #     y = (r * np.sqrt(i)) * np.sin(np.radians(angle))
    #     turtle.goto(x, y)
    #     turtle.dot(5 + i // 5)

    # r = 1
    # theta_max = 10 * 360
    # delta_theta = 10
    # theta = 0

    # turtle.penup()

    # while theta < theta_max:
    #     # Convert polar to cartesian coordinates
    #     x = r * np.sqrt(theta) * np.cos(np.radians(theta))
    #     y = r * np.sqrt(theta) * np.sin(np.radians(theta))

    #     print(r)
    #     print(x, y)

    #     turtle.goto(x, y)
    #     turtle.pendown()

    #     # Increase radius for next point
    #     r += delta_theta
    #     theta += delta_theta

    length = 500
    turns = 5

    t.penup()   # Lift the pen up off the paper
    for i in range(-360 * turns, 360 * turns):  # Loop through the desired number of degrees
        r = length * np.sqrt(abs(i) / (360*turns))  # Calculate the radius at the current angle
        theta = np.radians(i)  # Convert the angle from degrees to radians
        if i < 0: theta = np.pi - theta
        xx = r * np.cos(theta) + x  # Calculate the x-coordinate
        yy = r * np.sin(theta) + y  # Calculate the y-coordinate
        t.goto(xx, yy)  # Move the turtle to the new location
        t.pendown()   # Put the pen back down on the paper

def spiral_in_circle_special_double_l(x, y, o, f, r, s=None):
    x, y, f, r = scale(x, y, f, r)

    reset(x, y, o)
    # # Setting the angle between the spiral arms
    # phi = 137.508

    # # Loop to draw the spiral
    # for i in range(200):
    #     angle = i * phi
    #     x = (r * np.sqrt(i)) * np.cos(np.radians(angle))
    #     y = (r * np.sqrt(i)) * np.sin(np.radians(angle))
    #     turtle.goto(x, y)
    #     turtle.dot(5 + i // 5)

    # r = 1
    # theta_max = 10 * 360
    # delta_theta = 10
    # theta = 0

    # turtle.penup()

    # while theta < theta_max:
    #     # Convert polar to cartesian coordinates
    #     x = r * np.sqrt(theta) * np.cos(np.radians(theta))
    #     y = r * np.sqrt(theta) * np.sin(np.radians(theta))

    #     print(r)
    #     print(x, y)

    #     turtle.goto(x, y)
    #     turtle.pendown()

    #     # Increase radius for next point
    #     r += delta_theta
    #     theta += delta_theta

    length = 500
    turns = 5

    t.penup()   # Lift the pen up off the paper
    for i in range(-360 * turns, 360 * turns):  # Loop through the desired number of degrees
        r = length * np.sqrt(abs(i) / (360*turns))  # Calculate the radius at the current angle
        theta = np.radians(i)  # Convert the angle from degrees to radians
        if i < 0: theta = np.pi - theta
        xx = r * np.cos(-theta) + x  # Calculate the x-coordinate
        yy = r * np.sin(-theta) + y  # Calculate the y-coordinate
        t.goto(xx, yy)  # Move the turtle to the new location
        t.pendown()   # Put the pen back down on the paper

def spiral_in_rounded_special_r(x, y, o, f, n, r, a, s=None):
    x, y, f, r = scale(x, y, f, r)

    reset(x, y, o)
    # d = 2
    # a = 0
    # t.penup()
    # t.goto(d * a / (2 * np.pi) * np.cos(np.radians(a)) + x, d * a / (2 * np.pi) * np.sin(np.radians(a)) + y)
    # t.pendown()
    # while a < 360 * 10:
    #     r = d * a / (2 * np.pi)
    #     t.goto(r * np.cos(np.radians(a)) + x, r * np.sin(np.radians(a)) + y)
    #     a += 1
    sign = 1
    if r < 0:
        t.right(180)
        sign = -1
        r = -r
    d = 0.9 * FACTOR
    for _ in range(10):
        if r < 0: break
        t.forward(r*np.pi/n)
        t.circle(sign*r, 180, steps=s)
        r -= d * FACTOR

        if r < 0: break
        t.forward(r*np.pi/n)
        t.circle(sign*r, 180, steps=s)
        r -= d * FACTOR

def spiral_in_rounded_special_l(x, y, o, f, n, r, a, s=None):
    x, y, f, r = scale(x, y, f, r)

    reset(x, y, o)
    sign = 1
    if r < 0:
        t.right(180)
        sign = -1
        r = -r
    d = 0.9 * FACTOR
    for _ in range(10):
        if r < 0: break
        t.forward(-r*np.pi/n)
        t.circle(sign*-r, -180, steps=s)
        r -= d * FACTOR

        if r < 0: break
        t.forward(-r*np.pi/n)
        t.circle(sign*-r, -180, steps=s)
        r -= d * FACTOR

# s.tracer(False)
# t.hideturtle()

t.pensize(2.5)

offset = 5
N_bound = +HEIGHT - offset
E_bound = +WIDTH - offset
S_bound = -HEIGHT + offset
W_bound = -WIDTH + offset

N_to_S = int(np.abs(np.diff([N_bound, S_bound]))[0])
E_to_W = int(np.abs(np.diff([E_bound, W_bound]))[0])

straight(-450, S_bound, 'y', N_to_S)
snake(-350-2.5*np.pi, S_bound, 'y', N_to_S, 5, 180)
mesh(-250+2.5*np.pi, S_bound, 'y', N_to_S, 5, 180)
slalom_special(-150, S_bound, 'y', N_to_S, 20, 180)
slalom_special_double(-50, S_bound, 'y', N_to_S, 20, 180)
# rounded(250, +250, 'y', 2, 5)
spiral_in_circle_special_double_r(125, -245, 'y', 10, 90)
spiral_in_circle_special_double_l(375, -245, 'y', 10, 90)
spiral_in_circle_special_r(75-2*np.pi-1, -425, 'y', 10, -100)
spiral_in_circle_special_l(175+2*np.pi+1, -425, 'y', 10, -100)
spiral_in_circle_special_r(325-2*np.pi-1, -425, 'y', 10, 100)
spiral_in_circle_special_l(425+2*np.pi+1, -425, 'y', 10, 100)
spiral_in_rounded_special_r(250, 270, 'y', 10, 2, 85, 180)
spiral_in_rounded_special_l(250, 270 +85*np.pi/2, 'y', 10, 2, 85, 180)
spiral_in_rounded_special_r(250, 90, 'y', 10, 2, -85, 180)
spiral_in_rounded_special_l(250, 90-85*np.pi/2, 'y', 10, 2, -85, 180)

# canvas = s.getcanvas()
# canvas.postscript(file='temp.ps')

# from PIL import Image

# img = Image.open('temp.ps')
# img.save('temp.png')

t.save_as('temp.svg')

from pyvips import Image

img = Image.new_from_file('temp.svg')
img.write_to_file('temp.png')

sys.exit()

# funktionen als komponenten implementieren
# random kombinieren von komponenten
# randomisieren der parameter pro iteration

# x^3 - 3x
# e-funktion, ln-funktion
# wellen funktionen
# algebraische kurven
# rotation, kreisrotation außen, kreisrotation innen






# X = -500
# Y = 0
# O = 'x'
X = 0
Y = -500
O = 'y'

L = 1000
R = 100
K = 100
D = 3
N = 3
A = 60

try: straight(X, Y, O, L) ; t.clear()
except: print(straight)
try: circle(X, Y, O, R) ; t.clear()
except: print(circle)
try: ellipsis(X, Y, O, R, R/2) ; t.clear()
except: print(ellipsis)
try: polygon(X, Y, O, K, D) ; t.clear()
except: print(polygon)

try: flower(X, Y, O, R, N, A) ; t.clear()
except: print(flower)
try: flower_poly(X, Y, O, K, N, A) ; t.clear()
except: print(flower_poly)
try: flower_inverse(X, Y, O, R, N, A) ; t.clear()
except: print(flower_inverse)
try: flower_inverse_poly(X, Y, O, K, N, A) ; t.clear()
except: print(flower_inverse_poly)

try: gear(X, Y, O, R, N) ; t.clear()
except: print(gear)
try: gear_poly(X, Y, O, K, N) ; t.clear()
except: print(gear_poly)
try: gear(X, Y, O, R, N) ; t.clear()
except: print(gear)
try: gear_poly(X, Y, O, K, N) ; t.clear()
except: print(gear_poly)

try: hills(X, Y, O, L, R, A) ; t.clear()
except: print(hills)
try: hills_poly(X, Y, O, L, K, A) ; t.clear()
except: print(hills_poly)
try: slalom(X, Y, O, L, R, A) ; t.clear()
except: print(slalom)
try: slalom_poly(X, Y, O, L, K, A) ; t.clear()
except: print(slalom_poly)

try: pos_signal(X, Y, O, L, R, A) ; t.clear()
except: print(pos_signal)
try: pos_signal_poly(X, Y, L, O, K, A) ; t.clear()
except: print(pos_signal_poly)
try: signal(X, Y, O, L, R, A) ; t.clear()
except: print(signal)
try: signal_poly(X, Y, O, L, K, A) ; t.clear()
except: print(signal_poly)

try: snake(X, Y, O, L, R, N) ; t.clear()
except: print(snake)
try: snake_poly(X, Y, O, L, K, A) ; t.clear()
except: print(snake_poly)

try: wave_lr(X, Y, O, L, R, A) ; t.clear()
except: print(wave_lr)
try: wave_rl(X, Y, O, L, R, A) ; t.clear()
except: print(wave_rl)
try: wave_lr_poly(X, Y, O, L, K, A) ; t.clear()
except: print(wave_lr_poly)
try: wave_rl_poly(X, Y, O, L, K, A) ; t.clear()
except: print(wave_rl_poly)

try: puzzle(X, Y, O, L, R, A) ; t.clear()
except: print(puzzle)
try: puzzle_poly(X, Y, O, L, K, A) ; t.clear()
except: print(puzzle_poly)
try: puzzle(X, Y, O, L, R, A) ; t.clear()
except: print(puzzle)
try: puzzle_poly(X, Y, O, L, K, A) ; t.clear()
except: print(puzzle_poly)

try: spiral_in_circle(X, Y, O, R, N) ; t.clear()
except: print(spiral_in_circle)
try: spiral_in_ellipsis(X, Y, O, R, N) ; t.clear()
except: print(spiral_in_ellipsis)
try: spiral_in_polygon(X, Y, O, R, N) ; t.clear()
except: print(spiral_in_polygon)

try: spiral_in_flower(X, Y, O, R, N) ; t.clear()
except: print(spiral_in_flower)
try: spiral_in_flower_poly(X, Y, O, R, N) ; t.clear()
except: print(spiral_in_flower_poly)

try: spiral_in_gear(X, Y, O, R, N) ; t.clear()
except: print(spiral_in_gear)
try: spiral_in_gear_poly(X, Y, O, R, N) ; t.clear()
except: print(spiral_in_gear_poly)

try: spiral_out_circle(X, Y, O, R, N) ; t.clear()
except: print(spiral_circle)
try: spiral_out_ellipsis(X, Y, O, R, N) ; t.clear()
except: print(spiral_ellipsis)
try: spiral_out_polygon(X, Y, O, R, N) ; t.clear()
except: print(spiral_polygon)

try: spiral_out_flower(X, Y, O, R, N) ; t.clear()
except: print(spiral_flower)
try: spiral_out_flower_poly(X, Y, O, R, N) ; t.clear()
except: print(spiral_flower_poly)

try: spiral_out_gear(X, Y, O, R, N) ; t.clear()
except: print(spiral_gear)
try: spiral_out_gear_poly(X, Y, O, R, N) ; t.clear()
except: print(spiral_gear_poly)




















if sys.argv[1] == 'draw':
    t.shape('turtle')
    t.speed('fastest')

    animate(False)

    if DEBUG:
        grid(1)
        animate(True)

    straight(-40, -20, 40)
    straight(-39.5, -20, 40)
    straight(-39, -20, 40)

    circle(-34, 10, 2)
    circle(-35, 10, 1)
    circle(-35.5, 10, 0.5)

    #hills(-34, 0, 40, 2)
    up_n_down(-34, 0, 40, 2)
    polygon_1(-34, 0, 5, 3)
    polygon_2(-34, 0, 5, 5)
    # circle_spiral(-34, 0, 2)

    zero(-34, -10, 2)
    zero(-35, -10, 1)
    zero(-35.5, -10, 0.5)

    slalom(-31, -20, 40, 2)
    slalom(-27.5, -20, 40, 1)
    slalom(-25.5, -20, 40, 0.5)

    intestine(-22, -20, 40, 2)
    intestine(-12.5, -20, 40, 1)
    intestine(-7.5, -20, 40, 0.5)

    snake_1(1.5, -20, 40, 2)
    snake_1(5.5, -20, 40, 1)
    snake_1(8, -20, 40, 0.5)

    snake_2(10, -20, 40, 2)
    snake_2(16.5, -20, 40, 1)
    snake_2(20, -20, 40, 0.5)

    puzzle(22.5, -20, 40, 2)
    puzzle(31, -20, 40, 1)
    puzzle(35.5, -20, 40, 0.5)

    ziczac(39.5, -20, 40, 1.25, 90)

    s.exitonclick()
elif sys.argv[1] == 'svg':
    tracks = [
        'straight_1', 'straight_2', 'straight_3',
        'curve_1', 'curve_2', 'curve_3',
        'zero_1', 'zero_2', 'zero_3',
        'slalom_1', 'slalom_2', 'slalom_3',
        'intestine_1', 'intestine_2', 'intestine_3',
        'snake_1_1', 'snake_1_2', 'snake_1_3',
        'snake_2_1', 'snake_2_2', 'snake_2_3',
        'puzzle_1', 'puzzle_2', 'puzzle_3',
        'ziczac'
    ]

    with open('model.xml', 'w') as fp:
        fp.write('')

    with open('road.xml', 'w') as fp:
        fp.write('')

    for i, name in enumerate(tracks):
        t.reset()

        if i == 0: straight(-40, -20, 40)
        elif i == 1: straight(-39.5, -20, 40)
        elif i == 2: straight(-39, -20, 40)

        elif i == 3: circle(-34, 10, 2)
        elif i == 4: circle(-35, 10, 1)
        elif i == 5: circle(-35.5, 10, 0.5)

        elif i == 6: zero(-34, -10, 2)
        elif i == 7: zero(-35, -10, 1)
        elif i == 8: zero(-35.5, -10, 0.5)

        elif i == 9: slalom(-31, -20, 40, 2)
        elif i == 10: slalom(-27.5, -20, 40, 1)
        elif i == 11: slalom(-25.5, -20, 40, 0.5)

        elif i == 12: intestine(-22, -20, 40, 2)
        elif i == 13: intestine(-12.5, -20, 40, 1)
        elif i == 14: intestine(-7.5, -20, 40, 0.5)

        elif i == 15: snake_1(1.5, -20, 40, 2)
        elif i == 16: snake_1(5.5, -20, 40, 1)
        elif i == 17: snake_1(8, -20, 40, 0.5)

        elif i == 18: snake_2(10, -20, 40, 2)
        elif i == 19: snake_2(16.5, -20, 40, 1)
        elif i == 20: snake_2(20, -20, 40, 0.5)

        elif i == 21: puzzle(22.5, -20, 40, 2)
        elif i == 22: puzzle(31, -20, 40, 1)
        elif i == 23: puzzle(35.5, -20, 40, 0.5)

        elif i == 24: ziczac(39.5, -20, 40, 1.25, 90)

        t.save_as('temp.svg')

        with open('temp.svg', 'r') as fp:
            ctx = fp.read()

        matches = re.findall('points="(.*?)"', ctx)
        matches = matches[-1:] + matches[:-1]

        tmp = []
        for match in matches:
            entries = match.split(' ')
            for entry in entries:
                x, y = entry.split(',')
                p_1 = round((float(y) / FACTOR) - HEIGHT, 6)
                p_2 = round((float(x) / FACTOR) - WIDTH, 6)
                p_3 = 0.001
                tmp.append((p_1, p_2, p_3))

        # TODO: remove duplicates only if they appear side by side?!
        # TODO: generate spawn points systematically by def of sequences within the first period

        with open('model.xml', 'a') as fp:
            point = tmp[0] # -1
            ctx = '<link name="{}">\n'.format(name)
            ctx += '\t<pose>{} {} 0 0 0 0</pose>\n'.format(point[0], point[1])
            ctx += '</link>\n'

            fp.write(ctx)

        with open('road.xml', 'a') as fp:
            ctx = '<road name="{}">\n'.format(name)
            ctx += '\t<width>0.025</width>\n'
            for point in tmp:
                ctx += '\t<point>{} {} {}</point>\n'.format(*point)
            ctx += '</road>\n'

            fp.write(ctx)
