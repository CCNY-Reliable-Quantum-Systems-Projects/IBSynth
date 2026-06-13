"""Used for generating inclusions for interblock (which blocks will be the starting point for interblock?)"""
def gen_incl_point_func(point_to_include):
    def incl_func(point):
        if point in point_to_include:
            return True
        else:
            return False
    return incl_func


def inclusion_function_even(circuitpoint):
    if circuitpoint[0] % 2 == 0:
        return True
    else:
        return False


def inclusion_function_odd(circuitpoint):
    if circuitpoint[0] % 2 == 0:
        return False
    else:
        return True


def inclusion_all(circpoint):
    return True


def inclusion_none(circpoint):
    return False


def make_point_inclusion(points):
    def point_inclusion(point):
        if point in points:
            return True
        else:
            return False

    return point_inclusion
