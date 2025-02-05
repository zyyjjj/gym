from gym.envs.mujoco import mujoco_env
from gym import utils
import numpy as np
from gym.envs.mujoco.humanoid import mass_center

class HumanoidStandupEnv(mujoco_env.MujocoEnv, utils.EzPickle):
    def __init__(self):
        mujoco_env.MujocoEnv.__init__(self, 'humanoidstandup.xml', 5)
        utils.EzPickle.__init__(self)
        
        print('body mass', self.model.body_mass)

    def _get_obs(self):
        data = self.sim.data
        return np.concatenate([data.qpos.flat[2:],
                               data.qvel.flat,
                               data.cinert.flat,
                               data.cvel.flat,
                               data.qfrc_actuator.flat,
                               data.cfrc_ext.flat])

    def step(self, a):
        pos_before = self.sim.data.qpos[2]
        self.do_simulation(a, self.frame_skip)
        pos_after = self.sim.data.qpos[2]
        data = self.sim.data
        uph_cost = (pos_after - 0) / self.model.opt.timestep
        up_vel = (pos_after - pos_before) / self.model.opt.timestep

        quad_ctrl_cost = 0.1 * np.square(data.ctrl).sum()
        quad_impact_cost = .5e-6 * np.square(data.cfrc_ext).sum()
        quad_impact_cost = min(quad_impact_cost, 10)
        reward = uph_cost - quad_ctrl_cost - quad_impact_cost + 1
        #print(reward)
        subtask_1_reward = mass_center(self.model, self.sim)
        #print('subtask_1_reward', mass_center(self.model, self.sim))
        subtask_2_reward = up_vel

        done = bool(False)
        return self._get_obs(), reward, done, \
            dict(reward_linup=uph_cost, 
                 reward_quadctrl=-quad_ctrl_cost, 
                 reward_impact=-quad_impact_cost, 
                 subtask_1 = subtask_1_reward,
                 subtask_2 = subtask_2_reward)

    def reset_model(self):
        c = 0.01
        self.set_state(
            self.init_qpos + self.np_random.uniform(low=-c, high=c, size=self.model.nq),
            self.init_qvel + self.np_random.uniform(low=-c, high=c, size=self.model.nv,)
        )
        return self._get_obs()

    def viewer_setup(self):
        self.viewer.cam.trackbodyid = 1
        self.viewer.cam.distance = self.model.stat.extent * 1.0
        self.viewer.cam.lookat[2] = 0.8925
        self.viewer.cam.elevation = -20
