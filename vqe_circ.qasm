// Generated by QIBO 0.2.12
OPENQASM 2.0;
include "qelib1.inc";
qreg q[5];
ry(2.1893365819515664) q[0];
ry(-3.1656150653092263) q[1];
ry(0.9226852950961806) q[2];
ry(1.1726419143997926) q[3];
ry(0.9331997624367696) q[4];
rz(-1.3129016584230702) q[0];
rz(-1.5086759551527145) q[1];
rz(2.7274816442107506) q[2];
rz(-2.7537138155606726) q[3];
rz(-2.2400833157242452) q[4];
cz q[0],q[1];
cz q[2],q[3];
ry(1.2353342403600083) q[0];
ry(2.2934019427520953) q[1];
ry(-3.9723066497820954) q[2];
ry(-0.9180258469301705) q[3];
ry(2.2681391894169862) q[4];
rz(0.9069890174989773) q[0];
rz(0.019434017034872953) q[1];
rz(-0.1927761370968049) q[2];
rz(2.8145578039057852) q[3];
rz(0.6817690075927756) q[4];
cz q[1],q[2];
cz q[0],q[4];
ry(-0.7953111818577518) q[0];
ry(-2.6522281064026316) q[1];
ry(-1.1656097357621076) q[2];
ry(2.0351859534553474) q[3];
ry(0.7713117266577931) q[4];