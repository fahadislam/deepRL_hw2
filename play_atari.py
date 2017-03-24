import tensorflow as tf
import gym

sess = tf.Session()
env = gym.make('SpaceInvaders-v0')
env.reset()
env.render()

while True:
    action = input('Next action: ')
    _, rt, _, _ = env.step(action)
    tf.summary.FileWriter('tmp/', sess.graph)
    env.render()
