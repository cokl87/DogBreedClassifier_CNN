# -*- coding: utf-8 -*-

"""
models.py

created: 10:03 - 01.09.20
author: kornel
"""


from webapp import db


class Query(db.Model):
    __tablename__ = 'query'

    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(), nullable=True)
    species = db.Column(db.Integer, nullable=False)
    dogbreed_int = db.Column(db.Integer, nullable=True)
    dogbreed_name = db.Column(db.String(40), nullable=True)
    model = db.Column(db.String(16), nullable=False)

    probabilities = db.relationship('Probabilities', backref='query', lazy=True)

    def __init__(self, name, species, dogbreed_int, dogbreed_name, model):
        self.name = name
        self.species = species
        self.dogbreed_int = dogbreed_int
        self.dogbreed_name = dogbreed_name
        self.model = model

    def __repr__(self):
        return '<id {}>'.format(self.id)


class Probabilities(db.Model):
    __tablename__ = 'probabilities'

    id = db.Column(db.Integer, primary_key=True)
    query_id = db.Column(db.Integer, db.ForeignKey('query.id'), nullable=False)
    p_001 = db.Column(db.Float, nullable=False)
    p_002 = db.Column(db.Float, nullable=False)
    p_003 = db.Column(db.Float, nullable=False)
    p_004 = db.Column(db.Float, nullable=False)
    p_005 = db.Column(db.Float, nullable=False)
    p_006 = db.Column(db.Float, nullable=False)
    p_007 = db.Column(db.Float, nullable=False)
    p_008 = db.Column(db.Float, nullable=False)
    p_009 = db.Column(db.Float, nullable=False)
    p_010 = db.Column(db.Float, nullable=False)
    p_011 = db.Column(db.Float, nullable=False)
    p_012 = db.Column(db.Float, nullable=False)
    p_013 = db.Column(db.Float, nullable=False)
    p_014 = db.Column(db.Float, nullable=False)
    p_015 = db.Column(db.Float, nullable=False)
    p_016 = db.Column(db.Float, nullable=False)
    p_017 = db.Column(db.Float, nullable=False)
    p_018 = db.Column(db.Float, nullable=False)
    p_019 = db.Column(db.Float, nullable=False)
    p_020 = db.Column(db.Float, nullable=False)
    p_021 = db.Column(db.Float, nullable=False)
    p_022 = db.Column(db.Float, nullable=False)
    p_023 = db.Column(db.Float, nullable=False)
    p_024 = db.Column(db.Float, nullable=False)
    p_025 = db.Column(db.Float, nullable=False)
    p_026 = db.Column(db.Float, nullable=False)
    p_027 = db.Column(db.Float, nullable=False)
    p_028 = db.Column(db.Float, nullable=False)
    p_029 = db.Column(db.Float, nullable=False)
    p_030 = db.Column(db.Float, nullable=False)
    p_031 = db.Column(db.Float, nullable=False)
    p_032 = db.Column(db.Float, nullable=False)
    p_033 = db.Column(db.Float, nullable=False)
    p_034 = db.Column(db.Float, nullable=False)
    p_035 = db.Column(db.Float, nullable=False)
    p_036 = db.Column(db.Float, nullable=False)
    p_037 = db.Column(db.Float, nullable=False)
    p_038 = db.Column(db.Float, nullable=False)
    p_039 = db.Column(db.Float, nullable=False)
    p_040 = db.Column(db.Float, nullable=False)
    p_041 = db.Column(db.Float, nullable=False)
    p_042 = db.Column(db.Float, nullable=False)
    p_043 = db.Column(db.Float, nullable=False)
    p_044 = db.Column(db.Float, nullable=False)
    p_045 = db.Column(db.Float, nullable=False)
    p_046 = db.Column(db.Float, nullable=False)
    p_047 = db.Column(db.Float, nullable=False)
    p_048 = db.Column(db.Float, nullable=False)
    p_049 = db.Column(db.Float, nullable=False)
    p_050 = db.Column(db.Float, nullable=False)
    p_051 = db.Column(db.Float, nullable=False)
    p_052 = db.Column(db.Float, nullable=False)
    p_053 = db.Column(db.Float, nullable=False)
    p_054 = db.Column(db.Float, nullable=False)
    p_055 = db.Column(db.Float, nullable=False)
    p_056 = db.Column(db.Float, nullable=False)
    p_057 = db.Column(db.Float, nullable=False)
    p_058 = db.Column(db.Float, nullable=False)
    p_059 = db.Column(db.Float, nullable=False)
    p_060 = db.Column(db.Float, nullable=False)
    p_061 = db.Column(db.Float, nullable=False)
    p_062 = db.Column(db.Float, nullable=False)
    p_063 = db.Column(db.Float, nullable=False)
    p_064 = db.Column(db.Float, nullable=False)
    p_065 = db.Column(db.Float, nullable=False)
    p_066 = db.Column(db.Float, nullable=False)
    p_067 = db.Column(db.Float, nullable=False)
    p_068 = db.Column(db.Float, nullable=False)
    p_069 = db.Column(db.Float, nullable=False)
    p_070 = db.Column(db.Float, nullable=False)
    p_071 = db.Column(db.Float, nullable=False)
    p_072 = db.Column(db.Float, nullable=False)
    p_073 = db.Column(db.Float, nullable=False)
    p_074 = db.Column(db.Float, nullable=False)
    p_075 = db.Column(db.Float, nullable=False)
    p_076 = db.Column(db.Float, nullable=False)
    p_077 = db.Column(db.Float, nullable=False)
    p_078 = db.Column(db.Float, nullable=False)
    p_079 = db.Column(db.Float, nullable=False)
    p_080 = db.Column(db.Float, nullable=False)
    p_081 = db.Column(db.Float, nullable=False)
    p_082 = db.Column(db.Float, nullable=False)
    p_083 = db.Column(db.Float, nullable=False)
    p_084 = db.Column(db.Float, nullable=False)
    p_085 = db.Column(db.Float, nullable=False)
    p_086 = db.Column(db.Float, nullable=False)
    p_087 = db.Column(db.Float, nullable=False)
    p_088 = db.Column(db.Float, nullable=False)
    p_089 = db.Column(db.Float, nullable=False)
    p_090 = db.Column(db.Float, nullable=False)
    p_091 = db.Column(db.Float, nullable=False)
    p_092 = db.Column(db.Float, nullable=False)
    p_093 = db.Column(db.Float, nullable=False)
    p_094 = db.Column(db.Float, nullable=False)
    p_095 = db.Column(db.Float, nullable=False)
    p_096 = db.Column(db.Float, nullable=False)
    p_097 = db.Column(db.Float, nullable=False)
    p_098 = db.Column(db.Float, nullable=False)
    p_099 = db.Column(db.Float, nullable=False)
    p_100 = db.Column(db.Float, nullable=False)
    p_101 = db.Column(db.Float, nullable=False)
    p_102 = db.Column(db.Float, nullable=False)
    p_103 = db.Column(db.Float, nullable=False)
    p_104 = db.Column(db.Float, nullable=False)
    p_105 = db.Column(db.Float, nullable=False)
    p_106 = db.Column(db.Float, nullable=False)
    p_107 = db.Column(db.Float, nullable=False)
    p_108 = db.Column(db.Float, nullable=False)
    p_109 = db.Column(db.Float, nullable=False)
    p_110 = db.Column(db.Float, nullable=False)
    p_111 = db.Column(db.Float, nullable=False)
    p_112 = db.Column(db.Float, nullable=False)
    p_113 = db.Column(db.Float, nullable=False)
    p_114 = db.Column(db.Float, nullable=False)
    p_115 = db.Column(db.Float, nullable=False)
    p_116 = db.Column(db.Float, nullable=False)
    p_117 = db.Column(db.Float, nullable=False)
    p_118 = db.Column(db.Float, nullable=False)
    p_119 = db.Column(db.Float, nullable=False)
    p_120 = db.Column(db.Float, nullable=False)
    p_121 = db.Column(db.Float, nullable=False)
    p_122 = db.Column(db.Float, nullable=False)
    p_123 = db.Column(db.Float, nullable=False)
    p_124 = db.Column(db.Float, nullable=False)
    p_125 = db.Column(db.Float, nullable=False)
    p_126 = db.Column(db.Float, nullable=False)
    p_127 = db.Column(db.Float, nullable=False)
    p_128 = db.Column(db.Float, nullable=False)
    p_129 = db.Column(db.Float, nullable=False)
    p_130 = db.Column(db.Float, nullable=False)
    p_131 = db.Column(db.Float, nullable=False)
    p_132 = db.Column(db.Float, nullable=False)
    p_133 = db.Column(db.Float, nullable=False)

    def __init__(self, qid, probs):
        self.query_id = qid
        for att, val in zip(('p_%03i' % idx for idx in range(1, 134)), probs):
            setattr(self, att, val)

    def __repr__(self):
        return '<id {}>'.format(self.id)
