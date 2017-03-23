import gym

env = gym.make('SpaceInvaders-v0')
env.reset()
env.render()

while True:
    action = input('Next action: ')
    env.step(action)
    env.render()
