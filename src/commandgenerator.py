import re


class CommandModel:
    """ Generates commands based on command templates and entities """

    def __init__(self):
        self.template_cache = {}
        self.template_mapper = {
            '{d}': ['D', 'C'],
            '{f}': ['F'],
            '{s}': ['C', 'S'],
            '{o}': ['F', 'T'],
            '{x}': ['W']
        }

    def command_parser(self, cmd):
        """ parse the command into verb|entity|preposition|entity2 """
        mobj = re.search(r'([\w\-\{\} ]+) (in|with|on|into|from) ([\w\-\{\} ]+)', cmd)
        if mobj:
            base, preposition, entity2 = mobj.groups()
        else:
            base = cmd
            preposition, entity2 = '', ''

        parts = base.split()
        verb, entity = parts[0], ' '.join(parts[1:])
        return {'verb': verb, 'entity': entity, 'preposition': preposition,
                'entity2': entity2}

    def filter_templates(self, templates):
        """ preprocess the templates """
        cache_key = tuple(sorted(templates))
        if cache_key in self.template_cache:
            return self.template_cache[cache_key]

        keys = self.template_mapper.keys()
        tp = [cmd.replace('{oven}','oven').replace('{stove}','stove').replace('{toaster}','BBQ')
              for cmd in templates]
        tp = [self.command_parser(cmd) for cmd in tp]
        tp = [p for p in tp if '{' not in p['entity2'] or p['entity2'] in keys ]
        out = []
        for p in tp:
            if '{' in p['entity2']:
                p['entity2'] = ''
                p['preposition'] = ''
            if p['entity']:
                out.append('{} {} {} {}'.format(p['verb'], p['entity'], p['preposition'], p['entity2']).strip())

        self.template_cache[cache_key] = out
        return out

    def get_ent_types(self, cat):
        output = []
        for k, values in self.template_mapper.items():
            if cat in values:
                output.append(k)
        return sorted(output)

    def generate_all(self, entities, templates):
        """ generates candidate commands based on the the entities and
        command templates
        """
        templates = self.filter_templates(templates)
        output = []
        for ent, cat in entities:
            etyps = self.get_ent_types(cat)
            for tpl in templates:
                for etyp in etyps:
                    if etyp in tpl:
                        output.append(tpl.replace(etyp, ent))
        entity_names = [e for e,_ in entities]
        for ent in ['north', 'south', 'east', 'west']:
            if ent in entity_names:
                output.append('go {}'.format(ent))
        output.append('prepare meal')
        return list(set(output))
