import pandas as pd


class HTP():
    def __init__(self):
        self.results = []

        self.wincount = 0
        self.chimcount = 0
        self.roofcount = 0
        self.branchcount = 0

    def window(self, window_count, window_size):
        if self.wincount == 0:
            self.wincount = +1
            self.results.append('- 창은 대인관계에 대한 주관적인, 세상과 상호작용하는 능력 경험, 그리고 환경과 간접적인 접촉을 의미합니다.')

            # 창문 개수
            window_num = 3
            if window_count > window_num:
                self.results.append('- 타인과 관계 맺고자 하는 욕구를 지니고 있습니다.')

        # 창문 크기 설정
        window_min_size = 100 * 100
        if window_size < window_min_size:
            self.results.append('- 심리적인 접근 불가, 사람에 대한 관심 부족일 수 있습니다.')

    def roof(self, roof_size):
        if self.roofcount == 0:
            self.roofcount = +1
            self.results.append('- 지붕은 내적 공상 활동, 자신의 생각, 관념, 기억, 정신생활, 공상 영역을 의미합니다.')

        # 지붕 크기 설정
        roof_stan_size = 200 * 100

        if roof_size > roof_stan_size:
            self.results.append('- 내적 공상 과정 강조로 공상에 많이 몰두하고 있습니다.')

        elif roof_size < roof_stan_size:
            self.results.append('- 내적 인지 과정을 회피, 억제, 억압하고 있습니다.')

    def branch(self, branch_size):
        if self.branchcount == 0:
            branchcount = +1
            self.results.append('- 가지는 환경으로부터 만족을 추구하는 능력을 의미합니다.')

        # 가지 크기 설정
        branch_stan_size = 200 * 100

        if branch_size > branch_stan_size:
            self.results.append('- 성취동기나 포부가 높아 불안하여 과잉 보상적인 행동을 합니다.')

        elif branch_size > branch_stan_size:
            self.results.append('- 우유부단하고 불안합니다.')

    def eye(self, eye_size):
        eye_stan_size = 200 * 50

        if eye_size > eye_stan_size:
            self.results.append('- 사회적 의견에 대한 과민성과 외향성을 지니고 있습니다.')

        elif eye_size < eye_stan_size:
            self.results.append('- 내향적이고 자기도취가 있으며 관조적입니다.')

    def mouth(self, mouth_size):
        mouth_stan_size = 100 * 50

        if mouth_size > mouth_stan_size:
            self.results.append('- 불안감이 있습니다.')

        elif mouth_size < mouth_stan_size:
            self.results.append('- 우울감이 있습니다.')

    def get_result(self, pred):
        self.results = []

        boxes_df = pd.DataFrame(pred[0]["boxes"], columns=['box_x', 'box_y', 'box_w', 'box_h'])
        labels_df = pd.DataFrame(map(int, pred[0]["labels"]), columns=['labels'])
        scores_df = pd.DataFrame(pred[0]["scores"], columns=['scores'])

        df = pd.concat([boxes_df, labels_df, scores_df], axis=1)

        for idx in range(len(df)):

            # 1.집

            # 창문
            if df.at[idx, 'labels'] == 43:
                window_count = len(df.loc[df['labels'] == 43])
                window_size = df.at[idx, 'box_w'] * df.at[idx, 'box_h']
                self.window(window_count, window_size)

            # 굴뚝
            if df.at[idx, 'labels'] == 2:
                if self.chimcount == 0:
                    self.results.append('- 굴뚝은 가족 간의 애정과 교류를 나타냅니다.')
                    self.chimcount += 1

                chim_count = len(df.loc[df['labels'] == 2])
                if chim_count > 1:
                    self.results.append('- 성 충동에 대한 과도한 염려, 친근한 인간관계에 대한 과도한 염려일 수 있습니다')

            # 지붕
            if df.at[idx, 'labels'] == 40:
                roof_size = df.at[idx, 'box_w'] * df.at[idx, 'box_h']
                self.roof(roof_size)

            # 태양
            if df.at[idx, 'labels'] == 45:
                self.results.append('- 강력한 부모와 같은 자기 대상 존재를 갈망하고, 의존성이 있습니다.')

            # 울타리
            if df.at[idx, 'labels'] == 36:
                self.results.append('- 방어의 책략을 지니고 있습니다.')

            # ------------------------------------------

            # 2. 나무
            if df.at[idx, 'labels'] == 9:
                tree_size = df.at[idx, 'box_w'] * df.at[idx, 'box_h']
                if tree_size > 500 * 500:
                    self.results.append('- 공격성 성향과 지배 욕구를 지니고 있습니다.')

            # 가지
            if df.at[idx, 'labels'] == 0:
                branch_size = df.at[idx, 'box_w'] * df.at[idx, 'box_h']
                self.branch(branch_size)

            # 뿌리
            if df.at[idx, 'labels'] == 23:
                root_size = df.at[idx, 'box_w'] * df.at[idx, 'box_h']
                # 뿌리 기준 크기 설정
                root_stan_size = 200 * 100

                if root_size > root_stan_size:
                    self.results.append('현실 접촉을 과도하게 강조하거나 염려하는 상태입니다.')

            # 열매
            if df.at[idx, 'labels'] == 34:
                self.results.append('- 부가적인 것들을 자신의 주변에 두어 든든함을 얻을 수 있습니다.')

            # ------------------------------------------

            # 3. 사람

            # 코, 팔, 발 유무
            if df.at[idx, 'labels'] == 30:
                if len(df.loc[df['labels'] == 44]) == 0:
                    self.results.append('- 자신이 타인에게 어떻게 보일지 매우 예민하고 두려워 합니다.')

                if len(df.loc[df['labels'] == 46]) == 0:
                    self.results.append('- 환경에 대해 불만이 있습니다.')

                if len(df.loc[df['labels'] == 21]) == 0:
                    self.results.append('- 독립성이 결여 되어 있습니다.')

            # 눈
            if df.at[idx, 'labels'] == 12:
                eye_size = df.at[idx, 'box_w'] * df.at[idx, 'box_h']
                self.eye(eye_size)

            # 귀
            if df.at[idx, 'labels'] == 3:
                nose_size = df.at[idx, 'box_w'] * df.at[idx, 'box_h']

                if nose_size > 100 * 100:
                    self.results.append('- 상당한 관계 망상적 태도가 있습니다.')

            # 입
            if df.at[idx, 'labels'] == 37:
                mouth_size = df.at[idx, 'box_w'] * df.at[idx, 'box_h']
                self.mouth(mouth_size)

        return self.results
