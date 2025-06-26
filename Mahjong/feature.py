from agent import MahjongGBAgent
from collections import defaultdict
import numpy as np
from pdb import set_trace as stx
import copy 

try:
    from MahjongGB import MahjongFanCalculator
except:
    print('MahjongGB library required! Please visit https://github.com/ailab-pku/PyMahjongGB for more information.')
    raise ValueError

class FeatureAgent(MahjongGBAgent):
    
    '''
    observation: 6*4*9
        (men+quan+hand4)*4*9  ??
    action_mask: 235
        pass1+hu1+discard34+chi63(3*7)+peng34+gang34+angang34+bugang34
    '''
    
    ACT_SIZE = 235
    
    OFFSET_OBS = {
        'HAND' : 0
    }
    OFFSET_ACT = {
        'Pass' : 0,
        'Hu' : 1,
        'Play' : 2,
        'Chi' : 36,
        'Peng' : 99,
        'Gang' : 133,
        'AnGang' : 167,
        'BuGang' : 201
    }
    TILE_LIST = [
        *('W%d'%(i+1) for i in range(9)),
        *('T%d'%(i+1) for i in range(9)),
        *('B%d'%(i+1) for i in range(9)),
        *('F%d'%(i+1) for i in range(4)),
        *('J%d'%(i+1) for i in range(3))
    ]
    OFFSET_TILE = {c : i for i, c in enumerate(TILE_LIST)}

    obs_pos = {}
    for i,c in enumerate(['W', 'T', 'B']):
        for j in range(1, 10):
            obs_pos[f'{c}{j}'] = (i*2, j-1)
    for j in range(1, 5):
        obs_pos[f'F{j}'] = (6, j-1)
    for j in range(1, 4):
        obs_pos[f'J{j}'] = (6, j+3)

    def __init__(self, seatWind): # seatWind: 0-3, 座位号
        self.seatWind = seatWind
        self.packs = [[] for i in range(4)]
        self.history = [[] for i in range(4)]
        self.tileWall = [21] * 4
        self.shownTiles = defaultdict(int)
        self.wallLast = False # 是否已经摸到牌墙中的最后一张牌
        self.isAboutKong = False # 上一个动作是否是杠
        self.obs = np.zeros((8, 9), dtype=np.uint8)
    '''
    Wind 0..3
    Deal XX XX ...
    Player N Draw
    Player N Gang
    Player N(me) AnGang XX
    Player N(me) Play XX
    Player N(me) BuGang XX
    Player N(not me) Peng
    Player N(not me) Chi XX
    Player N(not me) AnGang
    
    Player N Hu
    Huang
    Player N Invalid
    Draw XX
    Player N(not me) Play XX
    Player N(not me) BuGang XX
    Player N(me) Peng
    Player N(me) Chi XX
    '''
    def request2obs(self, request): # here !!
        # print(f"player {self.seatWind} ------------")
        # print(f"Request : {request}")
        # if hasattr(self, 'hand'):
        #     print("Hand: ", self.hand)
        # if self.packs[0]:
        #     print(f"Player {self.seatWind}' packs: {self.packs[0]}")

        t = request.split()
        if t[0] == 'Wind':
            self.prevalentWind = int(t[1])
            return
        if t[0] == 'Deal':
            self.hand = t[1:]
            # print(self.hand, type(self.hand))
            self._hand_embedding_update()
            return
        if t[0] == 'Huang':
            self.valid = []
            return self._obs()
        if t[0] == 'Draw':
            # Available: Hu, Play, AnGang, BuGang
            self.tileWall[0] -= 1
            self.wallLast = self.tileWall[1] == 0
            tile = t[1]
            self.valid = []
            if self._check_mahjong(tile, isSelfDrawn = True, isAboutKong = self.isAboutKong):
                self.valid.append(self.OFFSET_ACT['Hu'])
            self.isAboutKong = False
            self.hand.append(tile)
            self._hand_embedding_update()
            for tile in set(self.hand):
                self.valid.append(self.OFFSET_ACT['Play'] + self.OFFSET_TILE[tile])
                if self.hand.count(tile) == 4 and not self.wallLast and self.tileWall[0] > 0:
                    self.valid.append(self.OFFSET_ACT['AnGang'] + self.OFFSET_TILE[tile])
            if not self.wallLast and self.tileWall[0] > 0:
                for packType, tile, offer in self.packs[0]:
                    if packType == 'PENG' and tile in self.hand:
                        self.valid.append(self.OFFSET_ACT['BuGang'] + self.OFFSET_TILE[tile])
            return self._obs()
        # Player N Invalid/Hu/Draw/Play/Chi/Peng/Gang/AnGang/BuGang XX
        p = (int(t[1]) + 4 - self.seatWind) % 4 # 将自己编号为0. 
        if t[2] == 'Draw':
            self.tileWall[p] -= 1
            self.wallLast = self.tileWall[(p + 1) % 4] == 0
            return
        if t[2] == 'Invalid':
            self.valid = []
            return self._obs()
        if t[2] == 'Hu':
            self.valid = []
            return self._obs()
        if t[2] == 'Play':
            self.tileFrom = p
            self.curTile = t[3] # 上一张打出的牌。可供接下来吃/碰/杠/胡
            self.shownTiles[self.curTile] += 1
            self.history[p].append(self.curTile)
            if p == 0:
                self.hand.remove(self.curTile)
                self._hand_embedding_update()
                return
            else:
                # Available: Hu/Gang/Peng/Chi/Pass
                self.valid = []
                if self._check_mahjong(self.curTile):
                    self.valid.append(self.OFFSET_ACT['Hu'])
                if not self.wallLast:
                    if self.hand.count(self.curTile) >= 2:
                        self.valid.append(self.OFFSET_ACT['Peng'] + self.OFFSET_TILE[self.curTile])
                        if self.hand.count(self.curTile) == 3 and self.tileWall[0]:
                            self.valid.append(self.OFFSET_ACT['Gang'] + self.OFFSET_TILE[self.curTile])
                    color = self.curTile[0]
                    if p == 3 and color in 'WTB':
                        num = int(self.curTile[1])
                        tmp = []
                        for i in range(-2, 3): tmp.append(color + str(num + i))
                        if tmp[0] in self.hand and tmp[1] in self.hand:
                            self.valid.append(self.OFFSET_ACT['Chi'] + 'WTB'.index(color) * 21 + (num - 3) * 3 + 2)
                        if tmp[1] in self.hand and tmp[3] in self.hand:
                            self.valid.append(self.OFFSET_ACT['Chi'] + 'WTB'.index(color) * 21 + (num - 2) * 3 + 1)
                        if tmp[3] in self.hand and tmp[4] in self.hand:
                            self.valid.append(self.OFFSET_ACT['Chi'] + 'WTB'.index(color) * 21 + (num - 1) * 3)
                self.valid.append(self.OFFSET_ACT['Pass'])
                return self._obs()
        if t[2] == 'Chi':
            # print(t, '-----------')
            tile = t[3] # [WTB 2-8]
            color = tile[0]
            num = int(tile[1])
            self.packs[p].append(('CHI', tile, int(self.curTile[1]) - num + 2)) # 中心tile:2-8 offset:1-3
            self.shownTiles[self.curTile] -= 1
            for i in range(-1, 2):
                self.shownTiles[color + str(num + i)] += 1
            self.wallLast = self.tileWall[(p + 1) % 4] == 0
            if p == 0:
                # Available: Play
                self.valid = []
                self.hand.append(self.curTile)
                for i in range(-1, 2):
                    self.hand.remove(color + str(num + i))
                self._hand_embedding_update()
                for tile in set(self.hand):
                    self.valid.append(self.OFFSET_ACT['Play'] + self.OFFSET_TILE[tile])
                return self._obs()
            else:
                return
        if t[2] == 'UnChi': # 没遇到. env中是按顺序吃/碰/杠的.
            tile = t[3]
            color = tile[0]
            num = int(tile[1])
            self.packs[p].pop()
            self.shownTiles[self.curTile] += 1
            for i in range(-1, 2):
                self.shownTiles[color + str(num + i)] -= 1
            if p == 0:
                for i in range(-1, 2):
                    self.hand.append(color + str(num + i))
                self.hand.remove(self.curTile)
                self._hand_embedding_update()
            return
        if t[2] == 'Peng':
            self.packs[p].append(('PENG', self.curTile, (4 + p - self.tileFrom) % 4)) # 不是chi中的offset,是座位号
            self.shownTiles[self.curTile] += 2
            self.wallLast = self.tileWall[(p + 1) % 4] == 0
            if p == 0:
                # Available: Play
                self.valid = []
                for i in range(2):
                    self.hand.remove(self.curTile)
                self._hand_embedding_update()
                for tile in set(self.hand):
                    self.valid.append(self.OFFSET_ACT['Play'] + self.OFFSET_TILE[tile])
                return self._obs()
            else:
                return
        if t[2] == 'UnPeng': # 没遇到
            self.packs[p].pop()
            self.shownTiles[self.curTile] -= 2
            if p == 0:
                for i in range(2):
                    self.hand.append(self.curTile)
                self._hand_embedding_update()
            return
        if t[2] == 'Gang':
            self.packs[p].append(('GANG', self.curTile, (4 + p - self.tileFrom) % 4))
            self.shownTiles[self.curTile] += 3
            if p == 0:
                for i in range(3):
                    self.hand.remove(self.curTile)
                self._hand_embedding_update()
                self.isAboutKong = True
            return
        if t[2] == 'AnGang':
            tile = 'CONCEALED' if p else t[3]
            self.packs[p].append(('GANG', tile, 0))
            if p == 0:
                self.isAboutKong = True
                for i in range(4):
                    self.hand.remove(tile)
            else:
                self.isAboutKong = False
            return
        if t[2] == 'BuGang':
            tile = t[3]
            for i in range(len(self.packs[p])):
                if tile == self.packs[p][i][1]:
                    self.packs[p][i] = ('GANG', tile, self.packs[p][i][2])
                    break
            self.shownTiles[tile] += 1
            if p == 0:
                self.hand.remove(tile)
                self._hand_embedding_update()
                self.isAboutKong = True
                return
            else:
                # Available: Hu/Pass
                self.valid = []
                if self._check_mahjong(tile, isSelfDrawn = False, isAboutKong = True):
                    self.valid.append(self.OFFSET_ACT['Hu'])
                self.valid.append(self.OFFSET_ACT['Pass'])
                return self._obs()
        raise NotImplementedError('Unknown request %s!' % request)
    
    def action2response(self, action):
        if action < self.OFFSET_ACT['Hu']:
            return 'Pass'
        if action < self.OFFSET_ACT['Play']:
            return 'Hu'
        if action < self.OFFSET_ACT['Chi']:
            return 'Play ' + self.TILE_LIST[action - self.OFFSET_ACT['Play']]
        if action < self.OFFSET_ACT['Peng']:
            t = (action - self.OFFSET_ACT['Chi']) // 3
            return 'Chi ' + 'WTB'[t // 7] + str(t % 7 + 2)
        if action < self.OFFSET_ACT['Gang']:
            return 'Peng'
        if action < self.OFFSET_ACT['AnGang']:
            return 'Gang'
        if action < self.OFFSET_ACT['BuGang']: # AnGang
            return 'Gang ' + self.TILE_LIST[action - self.OFFSET_ACT['AnGang']]
        return 'BuGang ' + self.TILE_LIST[action - self.OFFSET_ACT['BuGang']]
    
    def response2action(self, response):
        t = response.split()
        if t[0] == 'Pass': return self.OFFSET_ACT['Pass']
        if t[0] == 'Hu': return self.OFFSET_ACT['Hu']
        if t[0] == 'Play': return self.OFFSET_ACT['Play'] + self.OFFSET_TILE[t[1]]
        # if t[0] == 'Chi': return self.OFFSET_ACT['Chi'] + 'WTB'.index(t[1][0]) * 7 * 3 + (int(t[2][1]) - 2) * 3 + int(t[1][1]) - int(t[2][1]) + 1
        if t[0] == 'Chi': return self.OFFSET_ACT['Chi'] + 'WTB'.index(t[1][0]) * 21 + (int(t[1][1]) - 2) * 3
        if t[0] == 'Peng': return self.OFFSET_ACT['Peng'] + self.OFFSET_TILE[t[1]]
        if t[0] == 'Gang': return self.OFFSET_ACT['Gang'] + self.OFFSET_TILE[t[1]]
        if t[0] == 'AnGang': return self.OFFSET_ACT['AnGang'] + self.OFFSET_TILE[t[1]]
        if t[0] == 'BuGang': return self.OFFSET_ACT['BuGang'] + self.OFFSET_TILE[t[1]]
        return self.OFFSET_ACT['Pass']
    
    def _obs(self):
        mask = np.zeros(self.ACT_SIZE)
        for a in self.valid:
            mask[a] = 1
        return {
            'observation': self.obs.copy(), # here
            'action_mask': mask
        }
    
    def get_embedding(self, hand, pack):
        obs = np.zeros((8, 9), dtype=np.uint8)
        for tile in hand: # WTBFJ
            x, y = self.obs_pos[tile]
            obs[x][y] += 1
        for pack in pack:
            op, tile, offset = pack
            if op == 'CHI':
                for i in range(-1, 2):
                    ti = tile[0] + str(int(tile[1])+i)
                    x, y = self.obs_pos[ti]
                    obs[x+1][y] += 1
            elif op == 'PENG' or op == 'GANG':
                num = 3 if op == 'PENG' else 4
                x, y = self.obs_pos[tile]
                obs[x+1][y] += num
            else:
                raise NotImplementedError
        return obs

    def _hand_embedding_update(self): # here
        self.obs = self.get_embedding(self.hand, self.packs[0])
        
    def get_next_state(self, action):
        sim_hand = copy.deepcopy(self.hand)
        sim_packs_self = copy.deepcopy(self.packs[0]) 

        action_type = None
        action_tile_str = None 
        
        if action < self.OFFSET_ACT['Hu']: # Pass
            action_type = 'Pass'
            # No changes to hand or packs for a Pass action.
        elif action < self.OFFSET_ACT['Play']: # Hu
            action_type = 'Hu'
            if self.curTile: # If it's a Hu on another's discard
                 sim_hand.append(self.curTile) # Temporarily add for embedding calculation
            # No tiles removed for Hu, it's a terminal state transition.
        elif action < self.OFFSET_ACT['Chi']: # Play
            action_type = 'Play'
            action_tile_str = self.TILE_LIST[action - self.OFFSET_ACT['Play']]
            if action_tile_str in sim_hand:
                sim_hand.remove(action_tile_str)
            else:
                print(f"Warning: Attempted to play {action_tile_str}, but not in hand: {sim_hand}")
        elif action < self.OFFSET_ACT['Peng']: # Chi
            action_type = 'Chi'
            t = (action - self.OFFSET_ACT['Chi']) // 3
            color_char = 'WTB'[t // 7]
            center_tile_num = t % 7 + 2
        
            center_tile_str = f"{color_char}{center_tile_num}"
            chi_tiles_seq = [f"{color_char}{center_tile_num-1}", center_tile_str, f"{color_char}{center_tile_num+1}"]
            
            tiles_to_remove_from_hand = [t for t in chi_tiles_seq if t != self.curTile]
            for t_rem in tiles_to_remove_from_hand:
                if t_rem in sim_hand:
                    sim_hand.remove(t_rem)
                else:
                    print(f"Warning: Attempted to remove {t_rem} for Chi, but not in hand: {sim_hand}")
            sim_packs_self.append(('CHI', center_tile_str, self.curTile)) 

        elif action < self.OFFSET_ACT['AnGang']: # Peng or Gang (external from curTile)
            if action < self.OFFSET_ACT['Gang']: # Peng
                action_type = 'Peng'
                action_tile_str = self.TILE_LIST[action - self.OFFSET_ACT['Peng']]
                for _ in range(2): 
                    if action_tile_str in sim_hand:
                        sim_hand.remove(action_tile_str)
                    else:
                        print(f"Warning: Attempted to remove {action_tile_str} for Peng, but not enough in hand: {sim_hand}")
                sim_packs_self.append(('PENG', action_tile_str, self.curTile))
            else: # Gang (external from curTile)
                action_type = 'Gang'
                action_tile_str = self.TILE_LIST[action - self.OFFSET_ACT['Gang']]
                for _ in range(3): 
                    if action_tile_str in sim_hand:
                        sim_hand.remove(action_tile_str)
                    else:
                        print(f"Warning: Attempted to remove {action_tile_str} for Gang, but not enough in hand: {sim_hand}")
                sim_packs_self.append(('GANG', action_tile_str, self.curTile))

        elif action < self.OFFSET_ACT['BuGang']: # AnGang (concealed)
            action_type = 'AnGang'
            action_tile_str = self.TILE_LIST[action - self.OFFSET_ACT['AnGang']]
            for _ in range(4): 
                if action_tile_str in sim_hand:
                    sim_hand.remove(action_tile_str)
                else:
                    print(f"Warning: Attempted to remove {action_tile_str} for AnGang, but not enough in hand: {sim_hand}")
            sim_packs_self.append(('GANG', action_tile_str, 'CONCEALED'))

        else: # BuGang (add to existing Peng)
            action_type = 'BuGang'
            action_tile_str = self.TILE_LIST[action - self.OFFSET_ACT['BuGang']]
            if action_tile_str in sim_hand:
                sim_hand.remove(action_tile_str)
            else:
                print(f"Warning: Attempted to remove {action_tile_str} for BuGang, but not in hand: {sim_hand}")
            found_peng = False
            for i, (pack_type, tile_val, offer) in enumerate(sim_packs_self):
                if pack_type == 'PENG' and tile_val == action_tile_str:
                    sim_packs_self[i] = ('GANG', tile_val, offer)
                    found_peng = True
                    break
            if not found_peng:
                print(f"Warning: BuGang on {action_tile_str} failed, no existing PENG pack found.")
            
        next_obs = self.get_embedding(sim_hand, sim_packs_self)

        return next_obs
    
    def _check_mahjong(self, winTile, isSelfDrawn = False, isAboutKong = False):
        # print('----------------')
        try:
            fans = MahjongFanCalculator(
                pack = tuple(self.packs[0]),
                hand = tuple(self.hand),
                winTile = winTile,
                flowerCount = 0,
                isSelfDrawn = isSelfDrawn,
                is4thTile = (self.shownTiles[winTile] + isSelfDrawn) == 4,
                isAboutKong = isAboutKong,
                isWallLast = self.wallLast,
                seatWind = self.seatWind,
                prevalentWind = self.prevalentWind,
                verbose = True
            )
            # print(fans, '?????????')
            fanCnt = 0
            for fanPoint, cnt, fanName, fanNameEn in fans:
                fanCnt += fanPoint * cnt
            if fanCnt < 8: raise Exception('Not Enough Fans') # here
        except:
            return False
        return True


if __name__ == '__main__':
    print("--- Running get_next_state Tests ---")
    
    agent = FeatureAgent(seatWind=0)

    def print_obs(obs_array, title):
        print(f"\n--- {title} ---")
        print("Hand (W,T,B,F,J):")
        print(obs_array[0:1,:])
        print(obs_array[2:3,:])
        print(obs_array[4:5,:])
        print(obs_array[6:7,:])
        print("Packs (W,T,B,F,J):")
        print(obs_array[1:2,:])
        print(obs_array[3:4,:])
        print(obs_array[5:6,:])
        print(obs_array[7:8,:])
        print("-" * 20)

    # Test 1: Play action
    agent.hand = ['W1', 'W1', 'W1', 'W2', 'W3', 'W4', 'W5', 'T1', 'T2', 'T3', 'B1', 'B2', 'B3']
    agent._hand_embedding_update()
    print_obs(agent.obs, "Initial State for Play")
    action_play = agent.response2action('Play W1')
    next_obs_play = agent.get_next_state(action_play)
    print_obs(next_obs_play, "Next State after Playing W1")

    # Test 2: Peng action
    agent.hand = ['W1', 'W1', 'W2', 'W3', 'T1', 'T1', 'T2', 'T3', 'B1', 'B1', 'B2', 'B3']
    agent.curTile = 'W1'
    agent.tileFrom = 1
    agent._hand_embedding_update()
    print_obs(agent.obs, "Initial State for Peng")
    action_peng = agent.response2action('Peng W1')
    next_obs_peng = agent.get_next_state(action_peng)
    print_obs(next_obs_peng, "Next State after Peng W1")

    # Test 3: Chi action (W3,W4,W5 from W4 discarded)
    agent.hand = ['W2', 'W3', 'W5', 'W6', 'T1', 'T2', 'T3', 'B1', 'B2', 'B3', 'F1', 'F2']
    agent.curTile = 'W4'
    agent.tileFrom = 3
    agent._hand_embedding_update()
    print_obs(agent.obs, "Initial State for Chi")
    action_chi = agent.response2action('Chi W4')
    next_obs_chi = agent.get_next_state(action_chi)
    print_obs(next_obs_chi, "Next State after Chi W4 (Chi W3,W4,W5 with W4 discarded)")
    exit(0)

    # Test 4: Gang action (external from discard)
    agent.hand = ['F1', 'F1', 'F1', 'W1', 'W2', 'W3', 'T1', 'T2', 'T3', 'B1', 'B2', 'B3']
    agent.curTile = 'F1' # Another player discarded F1
    agent.tileFrom = 0 # Dummy player index
    agent._hand_embedding_update()
    print_obs(agent.obs, "Initial State for Gang (external)")
    action_gang_external = agent.response2action('Gang F1')
    next_obs_gang_external = agent.get_next_state(action_gang_external)
    print_obs(next_obs_gang_external, "Next State after Gang F1 (external)")
    # Expected: Three F1 removed from hand, four F1 (GANG) added to packs.

    # Test 5: AnGang action (concealed)
    agent.hand = ['B1', 'B1', 'B1', 'B1', 'T1', 'T2', 'T3', 'W1', 'W2', 'W3', 'F1', 'F2']
    agent._hand_embedding_update()
    print_obs(agent.obs, "Initial State for AnGang")
    action_angang = agent.response2action('AnGang B1')
    next_obs_angang = agent.get_next_state(action_angang)
    print_obs(next_obs_angang, "Next State after AnGang B1")

    # Test 6: BuGang action (upgrade Peng to Gang)
    agent.hand = ['T1', 'T2', 'T3', 'W1', 'W2', 'W3', 'F1', 'F2', 'B1']
    agent.packs[0] = [('PENG', 'B1', 2)]
    agent._hand_embedding_update()
    print_obs(agent.obs, "Initial State for BuGang")
    action_bugang = agent.response2action('BuGang B1')
    next_obs_bugang = agent.get_next_state(action_bugang)
    print_obs(next_obs_bugang, "Next State after BuGang B1")

    # Test 7: Hu action (on discard)
    agent.hand = ['T1', 'T1', 'T1', 'W1', 'W1', 'W1', 'B1', 'B2', 'B3', 'F1', 'F1', 'F1', 'F2']
    agent.curTile = 'W2'
    agent.tileFrom = 2
    agent._hand_embedding_update()
    print_obs(agent.obs, "Initial State for Hu (on discard)")
    action_hu = agent.response2action('Hu')
    next_obs_hu = agent.get_next_state(action_hu)
    print_obs(next_obs_hu, "Next State after Hu on W2")

    # Test 8: Pass action
    agent.hand = ['W1', 'W2', 'W3', 'T1', 'T2', 'T3', 'B1', 'B2', 'B3', 'F1', 'F2', 'F3', 'F4']
    agent.curTile = 'W9'
    agent.tileFrom = 1
    agent._hand_embedding_update()
    print_obs(agent.obs, "Initial State for Pass")
    action_pass = agent.response2action('Pass')
    next_obs_pass = agent.get_next_state(action_pass)
    print_obs(next_obs_pass, "Next State after Pass")