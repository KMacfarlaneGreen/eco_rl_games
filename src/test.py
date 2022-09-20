from ring_environment import Environment

env = Environment(100, 2, 'Test', 5000, 20)

map, agents, loc_to_agent, id_to_agent = env._generate_map

print('map', map, 'agents', agents, 'loc', loc_to_agent, 'id', id_to_agent)


