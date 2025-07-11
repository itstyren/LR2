class BaseScenario:  # defines scenario upon which the world is built
    def make_world(self):  # create elements of the world
        raise NotImplementedError()

    def reset_world(self, world, seed):  # create initial conditions of the world
        raise NotImplementedError()