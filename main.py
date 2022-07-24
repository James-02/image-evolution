#!/usr/bin/env python3
import sys
import random
import json
import numpy
import multiprocessing
import matplotlib.pyplot as plt

from math import atan2, pi
from deap import creator, base, tools, algorithms
from PIL import ImageDraw, Image, ImageChops

creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", list, fitness=creator.FitnessMax)

toolbox = base.Toolbox()

def read_config(PATH, TYPE):
    with open(PATH, "r") as f:
        data = json.load(f)
        return data[TYPE]

TARGET_IMAGE = read_config("config.json", "Image")["TARGET_IMAGE"]
MAX = 255 * 200 * 200
TARGET = Image.open(TARGET_IMAGE)
TARGET.load()  # read image and close the file

def sort_points(x, y):
    centroid = (sum(y) // len(y)), (sum(x) // len(x))
    points = []
    for x, y in zip(x, y):
        angle = atan2(centroid[0] - y, centroid[1] - x)
        points.append(((x, y), (angle) * 180 / pi))
    
    points.sort(key=lambda y: y[1])

    return points

def make_polygon(SMALL=0.8, LARGE=0.6):
    # 0 <= R|G|B < 256, 30 <= A <= 60, 10 <= x|y < 190
    R, G, B = random.sample(range(0, 235), 3)
    A = random.randint(30, 90)
    rand = random.random()

    if rand < LARGE:
        if version == "1":
            x1, y1, x2, y2, x3, y3, x4, y4, x5, y5 = random.sample(range(10, 190), 10)
            p = sort_points([x1, x2, x3, x4, x5], [y1, y2, y3, y4, y5])
            return [(R, G, B, A), (p[0][0]), (p[1][0]), (p[2][0]), (p[3][0]), (p[4][0])]
        
        else:
            x1, y1, x2, y2, x3, y3, x4, y4 = random.sample(range(10, 190), 8)
            p = sort_points([x1, x2, x3, x4], [y1, y2, y3, y4])
            return [(R, G, B, A), (p[0][0]), (p[1][0]), (p[2][0]), (p[3][0])]

    elif rand < SMALL:
        x1, y1 = random.sample(range(20, 150), 2)
        x2, y2 = random.sample(range(10, 30), 2)
        x3, y3 = random.sample(range(-20, 20), 2)

        x2 += x1
        x3 += x1
        y2 += y1
        y3 += y1
    
    else:
        if version == "1":
            x1, y1, x2, y2, x3, y3, x4, y4 = random.sample(range(10, 190), 8)
            p = sort_points([x1, x2, x3, x4], [y1, y2, y3, y4])
            return [(R, G, B, A), (p[0][0]), (p[1][0]), (p[2][0]), (p[3][0])]

        else:
            x1, y1, x2, y2, x3, y3 = random.sample(range(10, 190), 6)

    p = sort_points([x1, x2, x3], [y1, y2, y3])
    return [(R, G, B, A), (p[0][0]), (p[1][0]), (p[2][0])]


def mutate(solution, indpb):
    polygon = random.choice(solution)
    rand = random.random()
    if rand < 0.5:
        # mutate points
        coords = [x for point in polygon[1:] for x in point]
        tools.mutGaussian(coords, 0, 10, indpb)
        coords = [max(0, min(int(x), 200)) for x in coords]
        polygon[1:] = list(zip(coords[::2], coords[1::2]))
    
    elif rand < 0.6:
        x4, y4 = random.sample(range(0, 190), 2)
        idx = solution.index(polygon)
        points = sort_points([polygon[1][0], polygon[2][0], polygon[3][0], x4], 
                                [polygon[1][1], polygon[2][1], polygon[3][1], y4])

        polygon = [(polygon[0][0], polygon[0][1], polygon[0][2], polygon[0][3]), 
                 (points[0][0]), (points[1][0]), (points[2][0]), (points[3][0])]

        solution[idx] = polygon

    elif rand < 0.8:
        R, G, B = random.sample(range(-20, 20), 3)
        A = random.randint(-20, 20)
        p = polygon[0]
        polygon[0] = (p[0]+ R, p[1] + G, p[2] + B, p[3] + A)

    else:
        if len(solution) < 28:
            solution.append(make_polygon(LARGE=0.8, SMALL=0.9))
        elif len(solution) < 40:
            solution.append(make_polygon(LARGE=0.4, SMALL=0.8))
        elif len(solution) < 100:
            solution.append(make_polygon(LARGE=0.2, SMALL=0.9))
        else: 
            tools.mutShuffleIndexes(solution, indpb)

    return solution,


def draw(solution):
    image = Image.new("RGB", (200, 200))
    canvas = ImageDraw.Draw(image, "RGBA")
    for polygon in solution:
        canvas.polygon(polygon[1:], fill=polygon[0])

    return image


def evaluate(solution):
    image = draw(solution) 
    diff = ImageChops.difference(image, TARGET)
    hist = diff.convert("L").histogram()
    count = sum(i * n for i, n in enumerate(hist))
    return (MAX - count) / MAX,


def draw_graph(log):
    fig, ax1 = plt.subplots()
    ax1.plot(log.select("gen"), log.select("median"), color='red', label="Median Fitness")
    ax2 = ax1.twinx()
    ax2.plot(log.select("gen"), log.select("std"), color='blue', label="Standard Deviation")
    fig.tight_layout()

    ax1.set_xlabel("Generation")
    ax1.set_ylabel("Median Fitness", color='red')
    for num in ax1.get_yticklabels():
        num.set_color("red")

    ax2.set_ylabel("Standard Deviation", color='blue')
    for num in ax2.get_yticklabels():
        num.set_color("blue")

    return plt


def run(GENERATIONS, POPULATION, MUTATION_PROB,
         TOURNAMENT_SIZE, CROSS_PROB, IND_PROB, POLYGONS, SEED):
    random.seed(SEED)

    toolbox.register("map", pool.map)
    toolbox.register("individual", tools.initRepeat, creator.Individual, make_polygon, n=POLYGONS)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register("evaluate", evaluate)
    toolbox.register("mate", tools.cxTwoPoint)
    toolbox.register("mutate", mutate, indpb=IND_PROB)

    if version == "3":
        toolbox.register("select", tools.selDoubleTournament, fitness_first=True, parsimony_size=1, fitness_size=9)
    else:
        toolbox.register("select", tools.selTournament, tournsize=TOURNAMENT_SIZE)

    population = toolbox.population(n=POPULATION)
    hof = tools.HallOfFame(3)
    stats = tools.Statistics(lambda x: x.fitness.values[0])
    stats.register("min", numpy.min)
    stats.register("max", numpy.max)
    stats.register("median", numpy.median)
    stats.register("std", numpy.std)

    if version == "2":
        population, log = algorithms.eaMuPlusLambda(population, toolbox, mu=(len(population) // 5) * 4, 
                        lambda_=len(population), cxpb=CROSS_PROB, mutpb=MUTATION_PROB, 
                        ngen=GENERATIONS, stats=stats, halloffame=hof, verbose=True)

    else:
        population, log = algorithms.eaSimple(population, toolbox, 
                        cxpb=CROSS_PROB, mutpb=MUTATION_PROB, ngen=GENERATIONS, 
                        stats=stats, halloffame=hof, verbose=True)

    image = draw(population[0])
    image.save("media/solution.png")

    graph = draw_graph(log)
    graph.show()

def print_err():
    """ Function to print the correct function usage,
        to be called upon an error.
    """
    print("Usage: python main.py <code>\n\
            0 - Reference algorithm\n\
            1 - Solution Representation\n\
            2 - Offspring Generation\n\
            3 - Selection Function\n\
            ")

if __name__ == "__main__":
    try:
        global version
        version = sys.argv[1]
        if version in ["0", "1", "2", "3"]:
            params = read_config("config.json", version)
            pool = multiprocessing.Pool()
            run(**params)
            pool.close()
        
        else:
            raise IndexError

    except IndexError:
        print_err()
        