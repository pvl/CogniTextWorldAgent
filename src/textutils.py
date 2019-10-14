import re

class CompactPreprocessor:
    """ Preprocessor that tries to reduce the number of tokens by removing punctuation
    """
    def convert(self, description, recipe, inventory, entities):
        if recipe != '':
            txt = self.inventory_text(inventory, entities) + ' ' + recipe + ' ' + description
        else:
            txt = self.inventory_text(inventory, entities) + ' missing recipe ' + description
        txt = re.sub(r'\n', ' ', txt)
        # convert names with hiffen with space
        txt = re.sub(r'(\w)\-(\w)', r'\1 \2', txt)
        # remove punctuation
        txt = re.sub(r'([.:\-!=#",?])', r' ', txt)
        txt = re.sub(r'\s{2,}', ' ', txt)
        return txt.strip('.')

    def inventory_text(self, inventory, entities):
        n_items = self.count_inventory_items(inventory, entities)
        text = '{} {}'.format(n_items, inventory)
        return text

    def count_inventory_items(self, inventory, entities):
        parts = [p.strip() for p in inventory.split('\n')]
        parts = [p for p in parts if p]
        return len([p for p in parts if any(p.find(ent) != -1 for ent in entities)])


# Adapted from NAIL Agent
class ConnectionGraph:
    def __init__(self):
        self._out_graph = {}
        self._in_graph  = {}

    def add_single(self, connection):
        """ Adds a new connection to the graph if it doesn't already exist. """
        from_location = connection.from_location
        to_location = connection.to_location
        if from_location in self._out_graph:
            if connection in self._out_graph[from_location]:
                return
            self._out_graph[from_location].append(connection)
        else:
            self._out_graph[from_location] = [connection]
        if to_location is not None:
            if to_location in self._in_graph:
                self._in_graph[to_location].append(connection)
            else:
                self._in_graph[to_location] = [connection]

    def add(self, connection):
        self.add_single(connection)
        action = self.revert_action(connection.action)
        rev = Connection(connection.to_location, action, connection.from_location)
        self.add_single(rev)

    def revert_action(self, action):
        revmap = {
            'go north': 'go south',
            'go south': 'go north',
            'go east': 'go west',
            'go west': 'go east'
        }
        return revmap[action]

    def incoming(self, location):
        """ Returns a list of incoming connections to the given location. """
        if location in self._in_graph:
            return self._in_graph[location]
        else:
            return []

    def outgoing(self, location):
        """ Returns a list of outgoing connections from the given location. """
        if location in self._out_graph:
            return self._out_graph[location]
        else:
            return []

    def known_exits(self, location):
        return [connection.action for connection in self.outgoing(location)]

    def reset_except_last(self, location):
        """ reset the list of know outgoing locations except the last one added """
        if location in self._out_graph:
            last = self._out_graph[location][-1]
            self._out_graph[location] = [last]


# Adapted from NAIL Agent
class Connection:
    def __init__(self, from_location, action, to_location):
        self.from_location = from_location
        self.to_location   = to_location
        self.action        = action

    def __eq__(self, other):
        if isinstance(self, other.__class__):
            return self.action == other.action and\
                self.from_location == other.from_location and\
                self.to_location == other.to_location
        return False

    def __str__(self):
        return "{} --({})--> {}".format(self.from_location,
                                        self.action,
                                        self.to_location)

    def __repr__(self):
        return str(self)
