import json
import math
import random
import sys

from mpi4py import MPI

LIGHT_SPEED_METRES_SECOND = 299792458
SECONDS_YEAR = 31540000
LIGHT_SPEED_METRES_YEAR = LIGHT_SPEED_METRES_SECOND * SECONDS_YEAR
GRAVITY = 6.674 * 10 ** -11
SOLAR_MASS = 2 * 10 ** 30

galaxy_object_count = 100
galaxy_width_light_years = 52000
galaxy_height_light_years = 52000
simulation_steps = 10000
step_interval_seconds = SECONDS_YEAR * 100000
initial_speed_metres_second = 100000
solar_mass_range = (SOLAR_MASS,
                    SOLAR_MASS * 10)
singularity_mass = SOLAR_MASS * 40000000000000000000000


def root_init():
    print("Rank size ", size)
    for i in range(1, size):
        data = i
        print("Testing rank", i, "with data block:", data)
        comm.send(data, dest=i, tag=i)
        recv_data = comm.recv(source=i,
                              tag=i)
        print("Received rank", i, "data block:", recv_data)

        if recv_data == i * 2:
            print("Rank success!")
        else:
            print("Rank error!")

    if rank == 0:
        print("All Ranks Tested")


def root_main():
    print("Generating", galaxy_object_count, "galactic objects")
    galactic_objects = generate_galaxy(galaxy_object_count, galaxy_width_light_years, galaxy_height_light_years)
    write_galaxy_file(0, galactic_objects)

    for step in range(1, simulation_steps):
        sys.stdout.flush()

        # divide galactic objects into tasks for each node
        data = [galactic_objects[x::size - 1] for x in range(size - 1)]

        for node in range(1, size):
            comm.send({'object_group': data[node - 1], 'galactic_objects': galactic_objects}, dest=node)

        galactic_objects = []
        for node in range(1, size):
            galactic_objects.extend(comm.recv(source=node))

        write_galaxy_file(step, galactic_objects)


def write_galaxy_file(step, galactic_objects):
    orbits = open('./orbit/orbits-' + str(step) + '.jsn', 'w')
    orbits.write(json.dumps(galactic_objects))
    orbits.close()


def node_init():
    data = comm.recv(source=0, tag=rank)
    data = data * 2
    comm.send(data, dest=0, tag=rank)


def node_main():
    while True:
        node_calculation()


def node_calculation():
    data = None
    data = comm.recv(data, source=0)
    galaxy_objects = data['galactic_objects']

    for galaxy_object in data['object_group']:

        movement_x = 0
        movement_y = 0
        object_one_mass = galaxy_object['mass']
        object_one_x = galaxy_object['x']
        object_one_y = galaxy_object['y']

        for object_neighbour in galaxy_objects:
            if object_neighbour['id'] is galaxy_object['id']:
                continue

            object_two_mass = object_neighbour['mass']
            object_two_x = object_neighbour['x']
            object_two_y = object_neighbour['y']

            distance = distance_between(object_one_x, object_one_y, object_two_x, object_two_y)

            if distance == 0:
                continue

            direction_x, direction_y = direction(object_one_x, object_one_y, object_two_x, object_two_y)
            direction_x, direction_y = direction_normalized(direction_x, direction_y, distance)

            force = gravity_force(distance, object_one_mass, object_two_mass)

            # adjust the force of the gravitation to the mass of the object
            movement_x += (direction_x * force) / object_one_mass
            movement_y += (direction_y * force) / object_one_mass

        # add orbital movement and adjust for time step.
        galaxy_object['xs'] += movement_x * step_interval_seconds
        galaxy_object['ys'] += movement_y * step_interval_seconds

        galaxy_object['xs'] = limit_speed(galaxy_object['xs'],
                                          -LIGHT_SPEED_METRES_SECOND * step_interval_seconds,
                                          LIGHT_SPEED_METRES_SECOND * step_interval_seconds)
        galaxy_object['ys'] = limit_speed(galaxy_object['ys'],
                                          -LIGHT_SPEED_METRES_SECOND * step_interval_seconds,
                                          LIGHT_SPEED_METRES_SECOND * step_interval_seconds)

        galaxy_object['x'] += galaxy_object['xs']
        galaxy_object['y'] += galaxy_object['ys']
    comm.send(data['object_group'], dest=0)


def direction_normalized(direction_x, direction_y, distance):
    return direction_x / distance, direction_y / distance


def limit_speed(value, min_value, max_value):
    return max(min(value, max_value), min_value)


def distance_to_light_years(distance):
    return distance * LIGHT_SPEED_METRES_YEAR


def gravity_force(distance, mass_one, mass_two):
    return (GRAVITY * mass_one * mass_two) / (distance ** 2)


def direction(mass_one_x, mass_one_y, mass_two_x, mass_two_y):
    return (mass_two_x - mass_one_x), (mass_two_y - mass_one_y)


def distance_between(mass_one_x, mass_one_y, mass_two_x, mass_two_y):
    return math.sqrt(((mass_two_x - mass_one_x) ** 2) + ((mass_two_y - mass_one_y) ** 2))


def generate_galaxy(galactic_objects_size, galactic_objects_width, galactic_objects_height):
    galactic_objects = [{'id': 0,
                         'type': 'singularity',
                         'x': (galactic_objects_width / 2) * LIGHT_SPEED_METRES_YEAR,
                         'y': (galactic_objects_width / 2) * LIGHT_SPEED_METRES_YEAR,
                         'xs': 0,
                         'ys': 0,
                         'mass': singularity_mass}]

    for i in range(1, galactic_objects_size):

        # get random direction
        x = (random.random() - 0.5) * 2
        y = (random.random() - 0.5) * 2
        # limit to circle
        x, y = direction_normalized(x, y, distance_between(0, 0, x, y))
        # scale to galaxy size
        x *= random.randrange(0, galactic_objects_width / 2) * LIGHT_SPEED_METRES_YEAR
        y *= random.randrange(0, galactic_objects_height / 2) * LIGHT_SPEED_METRES_YEAR
        x += (galactic_objects_width * LIGHT_SPEED_METRES_YEAR) / 2
        y += (galactic_objects_height * LIGHT_SPEED_METRES_YEAR) / 2

        # initial velocities, sets up a basic diamond shape for initial orbits, creating an interesting result in
        # later orbits steps
        xs = initial_speed_metres_second
        ys = -initial_speed_metres_second
        if x < (galactic_objects_width * LIGHT_SPEED_METRES_YEAR) / 2:
            ys = -ys
        if y < (galactic_objects_height * LIGHT_SPEED_METRES_YEAR) / 2:
            xs = -xs
        xs *= 1 - (x / (galactic_objects_width * LIGHT_SPEED_METRES_YEAR))
        ys *= 1 - (y / (galactic_objects_height * LIGHT_SPEED_METRES_YEAR))

        galactic_objects.append({'id': i,
                                 'type': 'star',
                                 'x': x,
                                 'y': y,
                                 'xs': xs * step_interval_seconds,
                                 'ys': ys * step_interval_seconds,
                                 'mass': random.randrange(solar_mass_range[0],
                                                          solar_mass_range[1])})

    return galactic_objects


def start():
    print("Rank", rank, "available")
    if rank == 0:
        root_init()
        root_main()
    if rank > 0:
        node_init()
        node_main()


if __name__ == "__main__":
    comm = MPI.COMM_WORLD
    size = comm.Get_size()
    rank = comm.Get_rank()
    start()
    exit
