# -*- coding: utf-8 -*-

"""
models.py

created: 10:03 - 01.09.20
author: kornel
"""


from webapp import db


class Query(db.Model):
    __tablename__ = 'queries'

    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(), nullable=True)
    species = db.Column(db.Integer, nullable=False)
    dogbreed_int = db.Column(db.Integer, nullable=False)
    dogbreed_name = db.Column(db.String(40), nullable=False)
    model = db.Column(db.String(16), nullable=False)

    probabilities = db.relationship('Probabilities', backref='query', lazy=True, nullable=True)

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
    p_001 = db.Columns(db.Float, nullable=False)
    p_002 = db.Columns(db.Float, nullable=False)
    p_003 = db.Columns(db.Float, nullable=False)
    p_004 = db.Columns(db.Float, nullable=False)
    p_005 = db.Columns(db.Float, nullable=False)
    p_006 = db.Columns(db.Float, nullable=False)
    p_007 = db.Columns(db.Float, nullable=False)
    p_008 = db.Columns(db.Float, nullable=False)
    p_009 = db.Columns(db.Float, nullable=False)
    p_010 = db.Columns(db.Float, nullable=False)
    p_011 = db.Columns(db.Float, nullable=False)
    p_012 = db.Columns(db.Float, nullable=False)
    p_013 = db.Columns(db.Float, nullable=False)
    p_014 = db.Columns(db.Float, nullable=False)
    p_015 = db.Columns(db.Float, nullable=False)
    p_016 = db.Columns(db.Float, nullable=False)
    p_017 = db.Columns(db.Float, nullable=False)
    p_018 = db.Columns(db.Float, nullable=False)
    p_019 = db.Columns(db.Float, nullable=False)
    p_020 = db.Columns(db.Float, nullable=False)
    p_021 = db.Columns(db.Float, nullable=False)
    p_022 = db.Columns(db.Float, nullable=False)
    p_023 = db.Columns(db.Float, nullable=False)
    p_024 = db.Columns(db.Float, nullable=False)
    p_025 = db.Columns(db.Float, nullable=False)
    p_026 = db.Columns(db.Float, nullable=False)
    p_027 = db.Columns(db.Float, nullable=False)
    p_028 = db.Columns(db.Float, nullable=False)
    p_029 = db.Columns(db.Float, nullable=False)
    p_030 = db.Columns(db.Float, nullable=False)
    p_031 = db.Columns(db.Float, nullable=False)
    p_032 = db.Columns(db.Float, nullable=False)
    p_033 = db.Columns(db.Float, nullable=False)
    p_034 = db.Columns(db.Float, nullable=False)
    p_035 = db.Columns(db.Float, nullable=False)
    p_036 = db.Columns(db.Float, nullable=False)
    p_037 = db.Columns(db.Float, nullable=False)
    p_038 = db.Columns(db.Float, nullable=False)
    p_039 = db.Columns(db.Float, nullable=False)
    p_040 = db.Columns(db.Float, nullable=False)
    p_041 = db.Columns(db.Float, nullable=False)
    p_042 = db.Columns(db.Float, nullable=False)
    p_043 = db.Columns(db.Float, nullable=False)
    p_044 = db.Columns(db.Float, nullable=False)
    p_045 = db.Columns(db.Float, nullable=False)
    p_046 = db.Columns(db.Float, nullable=False)
    p_047 = db.Columns(db.Float, nullable=False)
    p_048 = db.Columns(db.Float, nullable=False)
    p_049 = db.Columns(db.Float, nullable=False)
    p_050 = db.Columns(db.Float, nullable=False)
    p_051 = db.Columns(db.Float, nullable=False)
    p_052 = db.Columns(db.Float, nullable=False)
    p_053 = db.Columns(db.Float, nullable=False)
    p_054 = db.Columns(db.Float, nullable=False)
    p_055 = db.Columns(db.Float, nullable=False)
    p_056 = db.Columns(db.Float, nullable=False)
    p_057 = db.Columns(db.Float, nullable=False)
    p_058 = db.Columns(db.Float, nullable=False)
    p_059 = db.Columns(db.Float, nullable=False)
    p_060 = db.Columns(db.Float, nullable=False)
    p_061 = db.Columns(db.Float, nullable=False)
    p_062 = db.Columns(db.Float, nullable=False)
    p_063 = db.Columns(db.Float, nullable=False)
    p_064 = db.Columns(db.Float, nullable=False)
    p_065 = db.Columns(db.Float, nullable=False)
    p_066 = db.Columns(db.Float, nullable=False)
    p_067 = db.Columns(db.Float, nullable=False)
    p_068 = db.Columns(db.Float, nullable=False)
    p_069 = db.Columns(db.Float, nullable=False)
    p_070 = db.Columns(db.Float, nullable=False)
    p_071 = db.Columns(db.Float, nullable=False)
    p_072 = db.Columns(db.Float, nullable=False)
    p_073 = db.Columns(db.Float, nullable=False)
    p_074 = db.Columns(db.Float, nullable=False)
    p_075 = db.Columns(db.Float, nullable=False)
    p_076 = db.Columns(db.Float, nullable=False)
    p_077 = db.Columns(db.Float, nullable=False)
    p_078 = db.Columns(db.Float, nullable=False)
    p_079 = db.Columns(db.Float, nullable=False)
    p_080 = db.Columns(db.Float, nullable=False)
    p_081 = db.Columns(db.Float, nullable=False)
    p_082 = db.Columns(db.Float, nullable=False)
    p_083 = db.Columns(db.Float, nullable=False)
    p_084 = db.Columns(db.Float, nullable=False)
    p_085 = db.Columns(db.Float, nullable=False)
    p_086 = db.Columns(db.Float, nullable=False)
    p_087 = db.Columns(db.Float, nullable=False)
    p_088 = db.Columns(db.Float, nullable=False)
    p_089 = db.Columns(db.Float, nullable=False)
    p_090 = db.Columns(db.Float, nullable=False)
    p_091 = db.Columns(db.Float, nullable=False)
    p_092 = db.Columns(db.Float, nullable=False)
    p_093 = db.Columns(db.Float, nullable=False)
    p_094 = db.Columns(db.Float, nullable=False)
    p_095 = db.Columns(db.Float, nullable=False)
    p_096 = db.Columns(db.Float, nullable=False)
    p_097 = db.Columns(db.Float, nullable=False)
    p_098 = db.Columns(db.Float, nullable=False)
    p_099 = db.Columns(db.Float, nullable=False)
    p_100 = db.Columns(db.Float, nullable=False)
    p_101 = db.Columns(db.Float, nullable=False)
    p_102 = db.Columns(db.Float, nullable=False)
    p_103 = db.Columns(db.Float, nullable=False)
    p_104 = db.Columns(db.Float, nullable=False)
    p_105 = db.Columns(db.Float, nullable=False)
    p_106 = db.Columns(db.Float, nullable=False)
    p_107 = db.Columns(db.Float, nullable=False)
    p_108 = db.Columns(db.Float, nullable=False)
    p_109 = db.Columns(db.Float, nullable=False)
    p_110 = db.Columns(db.Float, nullable=False)
    p_111 = db.Columns(db.Float, nullable=False)
    p_112 = db.Columns(db.Float, nullable=False)
    p_113 = db.Columns(db.Float, nullable=False)
    p_114 = db.Columns(db.Float, nullable=False)
    p_115 = db.Columns(db.Float, nullable=False)
    p_116 = db.Columns(db.Float, nullable=False)
    p_117 = db.Columns(db.Float, nullable=False)
    p_118 = db.Columns(db.Float, nullable=False)
    p_119 = db.Columns(db.Float, nullable=False)
    p_120 = db.Columns(db.Float, nullable=False)
    p_121 = db.Columns(db.Float, nullable=False)
    p_122 = db.Columns(db.Float, nullable=False)
    p_123 = db.Columns(db.Float, nullable=False)
    p_124 = db.Columns(db.Float, nullable=False)
    p_125 = db.Columns(db.Float, nullable=False)
    p_126 = db.Columns(db.Float, nullable=False)
    p_127 = db.Columns(db.Float, nullable=False)
    p_128 = db.Columns(db.Float, nullable=False)
    p_129 = db.Columns(db.Float, nullable=False)
    p_130 = db.Columns(db.Float, nullable=False)
    p_131 = db.Columns(db.Float, nullable=False)
    p_132 = db.Columns(db.Float, nullable=False)
    p_133 = db.Columns(db.Float, nullable=False)

    def __init__(self, probs):
        for att, val in zip(('p_%03i' in range(1, 134)), probs):
            setattr(self, att, val)

    def __repr__(self):
        return '<id {}>'.format(self.id)
