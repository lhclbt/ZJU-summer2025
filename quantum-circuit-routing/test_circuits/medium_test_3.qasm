OPENQASM 3.0;
include "stdgates.inc";
bit[16] ans;
qubit[16] q;
U(2.588865794424887, -1.168104532357039, -0.7971524685638545) q[0];
U(1.7780257989479222, -2.6253978644933977, -2.7157522178460067) q[1];
cz q[0], q[1];
U(pi/2, 2*pi/5, -pi) q[0];
U(pi, -pi, 0) q[1];
cz q[1], q[0];
U(pi, -pi/2, -pi) q[1];
U(pi/2, 0, pi) q[0];
cz q[0], q[1];
U(1.099557428756428, 0, -pi/2) q[0];
U(pi/2, -pi, 1.0995574287564276) q[1];
cz q[0], q[1];
U(pi/2, 2*pi/5, -pi) q[0];
U(pi, -pi, 0) q[1];
cz q[1], q[0];
U(pi, -pi/2, -pi) q[1];
U(pi/2, 0, pi) q[0];
cz q[0], q[1];
U(0, 0, pi/2) q[0];
U(pi/2, 0, pi/2) q[1];
cz q[0], q[1];
U(pi/2, 2*pi/5, -pi) q[0];
U(pi, -pi, 0) q[1];
cz q[1], q[0];
U(pi, -pi/2, -pi) q[1];
U(pi/2, 0, pi) q[0];
cz q[0], q[1];
U(2.084821383478164, 2.450625334071656, 1.3351338634352352) q[0];
U(1.0107821915673239, 1.8131315386304223, -0.6566734307263116) q[1];
U(2.588865794424887, -1.168104532357039, -0.7971524685638545) q[2];
U(1.7780257989479222, -2.6253978644933977, -2.7157522178460067) q[3];
cz q[2], q[3];
U(pi/2, 2*pi/5, -pi) q[2];
U(pi, -pi, 0) q[3];
cz q[3], q[2];
U(pi/2, 0, pi) q[2];
U(pi, -pi/2, -pi) q[3];
cz q[2], q[3];
U(1.099557428756428, 0, -pi/2) q[2];
U(pi/2, -pi, 1.0995574287564276) q[3];
cz q[2], q[3];
U(pi/2, 2*pi/5, -pi) q[2];
U(pi, -pi, 0) q[3];
cz q[3], q[2];
U(pi/2, 0, pi) q[2];
U(pi, -pi/2, -pi) q[3];
cz q[2], q[3];
U(0, 0, pi/2) q[2];
U(pi/2, 0, pi/2) q[3];
cz q[2], q[3];
U(pi/2, 2*pi/5, -pi) q[2];
U(pi, -pi, 0) q[3];
cz q[3], q[2];
U(pi/2, 0, pi) q[2];
U(pi, -pi/2, -pi) q[3];
cz q[2], q[3];
U(2.084821383478164, 2.450625334071656, 1.3351338634352352) q[2];
cz q[1], q[2];
U(pi/2, 2*pi/5, -pi) q[1];
U(pi, -pi, 0) q[2];
cz q[2], q[1];
U(pi/2, 0, pi) q[1];
U(pi, -pi/2, -pi) q[2];
cz q[1], q[2];
U(1.099557428756428, 0, -pi/2) q[1];
U(pi/2, -pi, 1.0995574287564276) q[2];
cz q[1], q[2];
U(pi/2, 2*pi/5, -pi) q[1];
U(pi, -pi, 0) q[2];
cz q[2], q[1];
U(pi/2, 0, pi) q[1];
U(pi, -pi/2, -pi) q[2];
cz q[1], q[2];
U(0, 0, pi/2) q[1];
U(pi/2, 0, pi/2) q[2];
cz q[1], q[2];
U(pi/2, 2*pi/5, -pi) q[1];
U(pi, -pi, 0) q[2];
cz q[2], q[1];
U(pi/2, 0, pi) q[1];
U(pi, -pi/2, -pi) q[2];
cz q[1], q[2];
U(1.5268029081266796, -1.2019829776562736, -1.7782294264593075) q[1];
U(1.1248260385203375, -2.985381781521503, 2.3207042664208153) q[2];
U(1.0107821915673239, 1.8131315386304223, -0.6566734307263116) q[3];
U(2.588865794424887, -1.168104532357039, -0.7971524685638545) q[4];
U(1.7780257989479222, -2.6253978644933977, -2.7157522178460067) q[5];
cz q[4], q[5];
U(pi/2, 2*pi/5, -pi) q[4];
U(pi, -pi, 0) q[5];
cz q[5], q[4];
U(pi/2, 0, pi) q[4];
U(pi, -pi/2, -pi) q[5];
cz q[4], q[5];
U(1.099557428756428, 0, -pi/2) q[4];
U(pi/2, -pi, 1.0995574287564276) q[5];
cz q[4], q[5];
U(pi/2, 2*pi/5, -pi) q[4];
U(pi, -pi, 0) q[5];
cz q[5], q[4];
U(pi/2, 0, pi) q[4];
U(pi, -pi/2, -pi) q[5];
cz q[4], q[5];
U(0, 0, pi/2) q[4];
U(pi/2, 0, pi/2) q[5];
cz q[4], q[5];
U(pi/2, 2*pi/5, -pi) q[4];
U(pi, -pi, 0) q[5];
cz q[5], q[4];
U(pi/2, 0, pi) q[4];
U(pi, -pi/2, -pi) q[5];
cz q[4], q[5];
U(2.084821383478164, 2.450625334071656, 1.3351338634352352) q[4];
cz q[3], q[4];
U(pi/2, 2*pi/5, -pi) q[3];
U(pi, -pi, 0) q[4];
cz q[4], q[3];
U(pi/2, 0, pi) q[3];
U(pi, -pi/2, -pi) q[4];
cz q[3], q[4];
U(1.099557428756428, 0, -pi/2) q[3];
U(pi/2, -pi, 1.0995574287564276) q[4];
cz q[3], q[4];
U(pi/2, 2*pi/5, -pi) q[3];
U(pi, -pi, 0) q[4];
cz q[4], q[3];
U(pi/2, 0, pi) q[3];
U(pi, -pi/2, -pi) q[4];
cz q[3], q[4];
U(0, 0, pi/2) q[3];
U(pi/2, 0, pi/2) q[4];
cz q[3], q[4];
U(pi/2, 2*pi/5, -pi) q[3];
U(pi, -pi, 0) q[4];
cz q[4], q[3];
U(pi/2, 0, pi) q[3];
U(pi, -pi/2, -pi) q[4];
cz q[3], q[4];
U(1.5268029081266796, -1.2019829776562736, -1.7782294264593075) q[3];
cz q[2], q[3];
U(pi/2, pi/20, -pi) q[2];
U(pi, -pi, 0) q[3];
cz q[3], q[2];
U(pi/2, 0, pi) q[2];
U(pi, -pi/2, -pi) q[3];
cz q[2], q[3];
U(pi/20, -pi/2, pi/2) q[2];
U(pi/2, -3*pi/4, -4*pi/5) q[3];
cz q[2], q[3];
U(pi/2, pi/20, -pi) q[2];
U(pi, -pi, 0) q[3];
cz q[3], q[2];
U(pi/2, 0, pi) q[2];
U(pi, -pi/2, -pi) q[3];
cz q[2], q[3];
U(7*pi/10, -pi/4, -pi/2) q[2];
U(pi/2, -pi, -1.7278759594743862) q[3];
U(1.1248260385203375, -2.985381781521503, 2.3207042664208153) q[4];
U(1.0107821915673239, 1.8131315386304223, -0.6566734307263116) q[5];
U(2.588865794424887, -1.168104532357039, -0.7971524685638545) q[6];
U(1.7780257989479222, -2.6253978644933977, -2.7157522178460067) q[7];
cz q[6], q[7];
U(pi/2, 2*pi/5, -pi) q[6];
U(pi, -pi, 0) q[7];
cz q[7], q[6];
U(pi/2, 0, pi) q[6];
U(pi, -pi/2, -pi) q[7];
cz q[6], q[7];
U(1.099557428756428, 0, -pi/2) q[6];
U(pi/2, -pi, 1.0995574287564276) q[7];
cz q[6], q[7];
U(pi/2, 2*pi/5, -pi) q[6];
U(pi, -pi, 0) q[7];
cz q[7], q[6];
U(pi/2, 0, pi) q[6];
U(pi, -pi/2, -pi) q[7];
cz q[6], q[7];
U(0, 0, pi/2) q[6];
U(pi/2, 0, pi/2) q[7];
cz q[6], q[7];
U(pi/2, 2*pi/5, -pi) q[6];
U(pi, -pi, 0) q[7];
cz q[7], q[6];
U(pi/2, 0, pi) q[6];
U(pi, -pi/2, -pi) q[7];
cz q[6], q[7];
U(2.084821383478164, 2.450625334071656, 1.3351338634352352) q[6];
cz q[5], q[6];
U(pi/2, 2*pi/5, -pi) q[5];
U(pi, -pi, 0) q[6];
cz q[6], q[5];
U(pi/2, 0, pi) q[5];
U(pi, -pi/2, -pi) q[6];
cz q[5], q[6];
U(1.099557428756428, 0, -pi/2) q[5];
U(pi/2, -pi, 1.0995574287564276) q[6];
cz q[5], q[6];
U(pi/2, 2*pi/5, -pi) q[5];
U(pi, -pi, 0) q[6];
cz q[6], q[5];
U(pi/2, 0, pi) q[5];
U(pi, -pi/2, -pi) q[6];
cz q[5], q[6];
U(0, 0, pi/2) q[5];
U(pi/2, 0, pi/2) q[6];
cz q[5], q[6];
U(pi/2, 2*pi/5, -pi) q[5];
U(pi, -pi, 0) q[6];
cz q[6], q[5];
U(pi/2, 0, pi) q[5];
U(pi, -pi/2, -pi) q[6];
cz q[5], q[6];
U(1.5268029081266796, -1.2019829776562736, -1.7782294264593075) q[5];
cz q[4], q[5];
U(pi/2, pi/20, -pi) q[4];
U(pi, -pi, 0) q[5];
cz q[5], q[4];
U(pi/2, 0, pi) q[4];
U(pi, -pi/2, -pi) q[5];
cz q[4], q[5];
U(pi/20, -pi/2, pi/2) q[4];
U(pi/2, -3*pi/4, -4*pi/5) q[5];
cz q[4], q[5];
U(pi/2, pi/20, -pi) q[4];
U(pi, -pi, 0) q[5];
cz q[5], q[4];
U(pi/2, 0, pi) q[4];
U(pi, -pi/2, -pi) q[5];
cz q[4], q[5];
U(7*pi/10, -pi/4, -pi/2) q[4];
cz q[3], q[4];
U(pi/2, pi/20, -pi) q[3];
U(pi, -pi, 0) q[4];
cz q[4], q[3];
U(pi/2, 0, pi) q[3];
U(pi, -pi/2, -pi) q[4];
cz q[3], q[4];
U(pi/20, -pi/2, pi/2) q[3];
U(pi/2, -3*pi/4, -4*pi/5) q[4];
cz q[3], q[4];
U(pi/2, pi/20, -pi) q[3];
U(pi, -pi, 0) q[4];
cz q[4], q[3];
U(pi/2, 0, pi) q[3];
U(pi, -pi/2, -pi) q[4];
cz q[3], q[4];
U(1.3727085346750192, 2.129204585735234, 1.3593715260007677) q[3];
U(0.5527268591649064, 1.9734881212327533, 1.4254709992818135) q[4];
U(pi/2, -pi, -1.7278759594743862) q[5];
U(1.1248260385203375, -2.985381781521503, 2.3207042664208153) q[6];
U(1.0107821915673239, 1.8131315386304223, -0.6566734307263116) q[7];
U(2.588865794424887, -1.168104532357039, -0.7971524685638545) q[8];
U(1.7780257989479222, -2.6253978644933977, -2.7157522178460067) q[9];
cz q[8], q[9];
U(pi/2, 2*pi/5, -pi) q[8];
U(pi, -pi, 0) q[9];
cz q[9], q[8];
U(pi/2, 0, pi) q[8];
U(pi, -pi/2, -pi) q[9];
cz q[8], q[9];
U(1.099557428756428, 0, -pi/2) q[8];
U(pi/2, -pi, 1.0995574287564276) q[9];
cz q[8], q[9];
U(pi/2, 2*pi/5, -pi) q[8];
U(pi, -pi, 0) q[9];
cz q[9], q[8];
U(pi/2, 0, pi) q[8];
U(pi, -pi/2, -pi) q[9];
cz q[8], q[9];
U(0, 0, pi/2) q[8];
U(pi/2, 0, pi/2) q[9];
cz q[8], q[9];
U(pi/2, 2*pi/5, -pi) q[8];
U(pi, -pi, 0) q[9];
cz q[9], q[8];
U(pi/2, 0, pi) q[8];
U(pi, -pi/2, -pi) q[9];
cz q[8], q[9];
U(2.084821383478164, 2.450625334071656, 1.3351338634352352) q[8];
cz q[7], q[8];
U(pi/2, 2*pi/5, -pi) q[7];
U(pi, -pi, 0) q[8];
cz q[8], q[7];
U(pi/2, 0, pi) q[7];
U(pi, -pi/2, -pi) q[8];
cz q[7], q[8];
U(1.099557428756428, 0, -pi/2) q[7];
U(pi/2, -pi, 1.0995574287564276) q[8];
cz q[7], q[8];
U(pi/2, 2*pi/5, -pi) q[7];
U(pi, -pi, 0) q[8];
cz q[8], q[7];
U(pi/2, 0, pi) q[7];
U(pi, -pi/2, -pi) q[8];
cz q[7], q[8];
U(0, 0, pi/2) q[7];
U(pi/2, 0, pi/2) q[8];
cz q[7], q[8];
U(pi/2, 2*pi/5, -pi) q[7];
U(pi, -pi, 0) q[8];
cz q[8], q[7];
U(pi/2, 0, pi) q[7];
U(pi, -pi/2, -pi) q[8];
cz q[7], q[8];
U(1.5268029081266796, -1.2019829776562736, -1.7782294264593075) q[7];
cz q[6], q[7];
U(pi/2, pi/20, -pi) q[6];
U(pi, -pi, 0) q[7];
cz q[7], q[6];
U(pi/2, 0, pi) q[6];
U(pi, -pi/2, -pi) q[7];
cz q[6], q[7];
U(pi/20, -pi/2, pi/2) q[6];
U(pi/2, -3*pi/4, -4*pi/5) q[7];
cz q[6], q[7];
U(pi/2, pi/20, -pi) q[6];
U(pi, -pi, 0) q[7];
cz q[7], q[6];
U(pi/2, 0, pi) q[6];
U(pi, -pi/2, -pi) q[7];
cz q[6], q[7];
U(7*pi/10, -pi/4, -pi/2) q[6];
cz q[5], q[6];
U(pi/2, pi/20, -pi) q[5];
U(pi, -pi, 0) q[6];
cz q[6], q[5];
U(pi/2, 0, pi) q[5];
U(pi, -pi/2, -pi) q[6];
cz q[5], q[6];
U(pi/20, -pi/2, pi/2) q[5];
U(pi/2, -3*pi/4, -4*pi/5) q[6];
cz q[5], q[6];
U(pi/2, pi/20, -pi) q[5];
U(pi, -pi, 0) q[6];
cz q[6], q[5];
U(pi/2, 0, pi) q[5];
U(pi, -pi/2, -pi) q[6];
cz q[5], q[6];
U(1.3727085346750192, 2.129204585735234, 1.3593715260007677) q[5];
cz q[4], q[5];
U(pi/2, 2*pi/5, -pi) q[4];
U(pi, -pi, 0) q[5];
cz q[5], q[4];
U(pi/2, 0, pi) q[4];
U(pi, -pi/2, -pi) q[5];
cz q[4], q[5];
U(1.099557428756428, 0, -pi/2) q[4];
U(pi/2, -pi, 1.0995574287564276) q[5];
cz q[4], q[5];
U(pi/2, 2*pi/5, -pi) q[4];
U(pi, -pi, 0) q[5];
cz q[5], q[4];
U(pi/2, 0, pi) q[4];
U(pi, -pi/2, -pi) q[5];
cz q[4], q[5];
U(0, 0, pi/2) q[4];
U(pi/2, 0, pi/2) q[5];
cz q[4], q[5];
U(pi/2, 2*pi/5, -pi) q[4];
U(pi, -pi, 0) q[5];
cz q[5], q[4];
U(pi/2, 0, pi) q[4];
U(pi, -pi/2, -pi) q[5];
cz q[4], q[5];
U(2.084821383478164, 2.450625334071656, 1.3351338634352352) q[4];
U(1.0107821915673239, 1.8131315386304223, -0.6566734307263116) q[5];
U(0.5527268591649064, 1.9734881212327533, 1.4254709992818135) q[6];
U(pi/2, -pi, -1.7278759594743862) q[7];
U(1.1248260385203375, -2.985381781521503, 2.3207042664208153) q[8];
U(1.0107821915673239, 1.8131315386304223, -0.6566734307263116) q[9];
U(2.588865794424887, -1.168104532357039, -0.7971524685638545) q[10];
U(1.7780257989479222, -2.6253978644933977, -2.7157522178460067) q[11];
cz q[10], q[11];
U(pi/2, 2*pi/5, -pi) q[10];
U(pi, -pi, 0) q[11];
cz q[11], q[10];
U(pi/2, 0, pi) q[10];
U(pi, -pi/2, -pi) q[11];
cz q[10], q[11];
U(1.099557428756428, 0, -pi/2) q[10];
U(pi/2, -pi, 1.0995574287564276) q[11];
cz q[10], q[11];
U(pi/2, 2*pi/5, -pi) q[10];
U(pi, -pi, 0) q[11];
cz q[11], q[10];
U(pi/2, 0, pi) q[10];
U(pi, -pi/2, -pi) q[11];
cz q[10], q[11];
U(0, 0, pi/2) q[10];
U(pi/2, 0, pi/2) q[11];
cz q[10], q[11];
U(pi/2, 2*pi/5, -pi) q[10];
U(pi, -pi, 0) q[11];
cz q[11], q[10];
U(pi/2, 0, pi) q[10];
U(pi, -pi/2, -pi) q[11];
cz q[10], q[11];
U(2.084821383478164, 2.450625334071656, 1.3351338634352352) q[10];
cz q[9], q[10];
U(pi, -pi, 0) q[10];
U(1.0107821915673239, 1.8131315386304223, -0.6566734307263116) q[11];
U(pi/2, 2*pi/5, -pi) q[9];
cz q[10], q[9];
U(pi/2, 0, pi) q[9];
U(pi, -pi/2, -pi) q[10];
cz q[9], q[10];
U(pi/2, -pi, 1.0995574287564276) q[10];
U(1.099557428756428, 0, -pi/2) q[9];
cz q[9], q[10];
U(pi, -pi, 0) q[10];
U(pi/2, 2*pi/5, -pi) q[9];
cz q[10], q[9];
U(pi/2, 0, pi) q[9];
U(pi, -pi/2, -pi) q[10];
cz q[9], q[10];
U(pi/2, 0, pi/2) q[10];
U(0, 0, pi/2) q[9];
cz q[9], q[10];
U(pi, -pi, 0) q[10];
U(pi/2, 2*pi/5, -pi) q[9];
cz q[10], q[9];
U(pi/2, 0, pi) q[9];
U(pi, -pi/2, -pi) q[10];
cz q[9], q[10];
U(1.1248260385203375, -2.985381781521503, 2.3207042664208153) q[10];
U(1.5268029081266796, -1.2019829776562736, -1.7782294264593075) q[9];
cz q[8], q[9];
U(pi/2, pi/20, -pi) q[8];
U(pi, -pi, 0) q[9];
cz q[9], q[8];
U(pi/2, 0, pi) q[8];
U(pi, -pi/2, -pi) q[9];
cz q[8], q[9];
U(pi/20, -pi/2, pi/2) q[8];
U(pi/2, -3*pi/4, -4*pi/5) q[9];
cz q[8], q[9];
U(pi/2, pi/20, -pi) q[8];
U(pi, -pi, 0) q[9];
cz q[9], q[8];
U(pi/2, 0, pi) q[8];
U(pi, -pi/2, -pi) q[9];
cz q[8], q[9];
U(7*pi/10, -pi/4, -pi/2) q[8];
cz q[7], q[8];
U(pi/2, pi/20, -pi) q[7];
U(pi, -pi, 0) q[8];
cz q[8], q[7];
U(pi/2, 0, pi) q[7];
U(pi, -pi/2, -pi) q[8];
cz q[7], q[8];
U(pi/20, -pi/2, pi/2) q[7];
U(pi/2, -3*pi/4, -4*pi/5) q[8];
cz q[7], q[8];
U(pi/2, pi/20, -pi) q[7];
U(pi, -pi, 0) q[8];
cz q[8], q[7];
U(pi/2, 0, pi) q[7];
U(pi, -pi/2, -pi) q[8];
cz q[7], q[8];
U(1.3727085346750192, 2.129204585735234, 1.3593715260007677) q[7];
cz q[6], q[7];
U(pi/2, 2*pi/5, -pi) q[6];
U(pi, -pi, 0) q[7];
cz q[7], q[6];
U(pi/2, 0, pi) q[6];
U(pi, -pi/2, -pi) q[7];
cz q[6], q[7];
U(1.099557428756428, 0, -pi/2) q[6];
U(pi/2, -pi, 1.0995574287564276) q[7];
cz q[6], q[7];
U(pi/2, 2*pi/5, -pi) q[6];
U(pi, -pi, 0) q[7];
cz q[7], q[6];
U(pi/2, 0, pi) q[6];
U(pi, -pi/2, -pi) q[7];
cz q[6], q[7];
U(0, 0, pi/2) q[6];
U(pi/2, 0, pi/2) q[7];
cz q[6], q[7];
U(pi/2, 2*pi/5, -pi) q[6];
U(pi, -pi, 0) q[7];
cz q[7], q[6];
U(pi/2, 0, pi) q[6];
U(pi, -pi/2, -pi) q[7];
cz q[6], q[7];
U(2.084821383478164, 2.450625334071656, 1.3351338634352352) q[6];
cz q[5], q[6];
U(pi/2, 2*pi/5, -pi) q[5];
U(pi, -pi, 0) q[6];
cz q[6], q[5];
U(pi/2, 0, pi) q[5];
U(pi, -pi/2, -pi) q[6];
cz q[5], q[6];
U(1.099557428756428, 0, -pi/2) q[5];
U(pi/2, -pi, 1.0995574287564276) q[6];
cz q[5], q[6];
U(pi/2, 2*pi/5, -pi) q[5];
U(pi, -pi, 0) q[6];
cz q[6], q[5];
U(pi/2, 0, pi) q[5];
U(pi, -pi/2, -pi) q[6];
cz q[5], q[6];
U(0, 0, pi/2) q[5];
U(pi/2, 0, pi/2) q[6];
cz q[5], q[6];
U(pi/2, 2*pi/5, -pi) q[5];
U(pi, -pi, 0) q[6];
cz q[6], q[5];
U(pi/2, 0, pi) q[5];
U(pi, -pi/2, -pi) q[6];
cz q[5], q[6];
U(1.1546395978862831, 1.6188980331309324, 2.9147019705055026) q[5];
U(2.67079632678808, 1.0999999999931829, 2.6707963267880794) q[6];
U(1.0107821915673239, 1.8131315386304223, -0.6566734307263116) q[7];
U(0.5527268591649064, 1.9734881212327533, 1.4254709992818135) q[8];
U(pi/2, -pi, -1.7278759594743862) q[9];
U(2.588865794424887, -1.168104532357039, -0.7971524685638545) q[12];
U(1.7780257989479222, -2.6253978644933977, -2.7157522178460067) q[13];
cz q[12], q[13];
U(pi/2, 2*pi/5, -pi) q[12];
U(pi, -pi, 0) q[13];
cz q[13], q[12];
U(pi/2, 0, pi) q[12];
U(pi, -pi/2, -pi) q[13];
cz q[12], q[13];
U(1.099557428756428, 0, -pi/2) q[12];
U(pi/2, -pi, 1.0995574287564276) q[13];
cz q[12], q[13];
U(pi/2, 2*pi/5, -pi) q[12];
U(pi, -pi, 0) q[13];
cz q[13], q[12];
U(pi/2, 0, pi) q[12];
U(pi, -pi/2, -pi) q[13];
cz q[12], q[13];
U(0, 0, pi/2) q[12];
U(pi/2, 0, pi/2) q[13];
cz q[12], q[13];
U(pi/2, 2*pi/5, -pi) q[12];
U(pi, -pi, 0) q[13];
cz q[13], q[12];
U(pi/2, 0, pi) q[12];
U(pi, -pi/2, -pi) q[13];
cz q[12], q[13];
U(2.084821383478164, 2.450625334071656, 1.3351338634352352) q[12];
cz q[11], q[12];
U(pi/2, 2*pi/5, -pi) q[11];
U(pi, -pi, 0) q[12];
cz q[12], q[11];
U(pi/2, 0, pi) q[11];
U(pi, -pi/2, -pi) q[12];
cz q[11], q[12];
U(1.099557428756428, 0, -pi/2) q[11];
U(pi/2, -pi, 1.0995574287564276) q[12];
cz q[11], q[12];
U(pi/2, 2*pi/5, -pi) q[11];
U(pi, -pi, 0) q[12];
cz q[12], q[11];
U(pi/2, 0, pi) q[11];
U(pi, -pi/2, -pi) q[12];
cz q[11], q[12];
U(0, 0, pi/2) q[11];
U(pi/2, 0, pi/2) q[12];
cz q[11], q[12];
U(pi/2, 2*pi/5, -pi) q[11];
U(pi, -pi, 0) q[12];
cz q[12], q[11];
U(pi/2, 0, pi) q[11];
U(pi, -pi/2, -pi) q[12];
cz q[11], q[12];
U(1.5268029081266796, -1.2019829776562736, -1.7782294264593075) q[11];
cz q[10], q[11];
U(pi/2, pi/20, -pi) q[10];
U(pi, -pi, 0) q[11];
cz q[11], q[10];
U(pi/2, 0, pi) q[10];
U(pi, -pi/2, -pi) q[11];
cz q[10], q[11];
U(pi/20, -pi/2, pi/2) q[10];
U(pi/2, -3*pi/4, -4*pi/5) q[11];
cz q[10], q[11];
U(pi/2, pi/20, -pi) q[10];
U(pi, -pi, 0) q[11];
cz q[11], q[10];
U(pi/2, 0, pi) q[10];
U(pi, -pi/2, -pi) q[11];
cz q[10], q[11];
U(7*pi/10, -pi/4, -pi/2) q[10];
cz q[9], q[10];
U(pi, -pi, 0) q[10];
U(pi/2, -pi, -1.7278759594743862) q[11];
U(1.1248260385203375, -2.985381781521503, 2.3207042664208153) q[12];
U(1.0107821915673239, 1.8131315386304223, -0.6566734307263116) q[13];
U(pi/2, pi/20, -pi) q[9];
cz q[10], q[9];
U(pi/2, 0, pi) q[9];
U(pi, -pi/2, -pi) q[10];
cz q[9], q[10];
U(pi/2, -3*pi/4, -4*pi/5) q[10];
U(pi/20, -pi/2, pi/2) q[9];
cz q[9], q[10];
U(pi, -pi, 0) q[10];
U(pi/2, pi/20, -pi) q[9];
cz q[10], q[9];
U(pi/2, 0, pi) q[9];
U(pi, -pi/2, -pi) q[10];
cz q[9], q[10];
U(0.5527268591649064, 1.9734881212327533, 1.4254709992818135) q[10];
U(1.3727085346750192, 2.129204585735234, 1.3593715260007677) q[9];
cz q[8], q[9];
U(pi/2, 2*pi/5, -pi) q[8];
U(pi, -pi, 0) q[9];
cz q[9], q[8];
U(pi/2, 0, pi) q[8];
U(pi, -pi/2, -pi) q[9];
cz q[8], q[9];
U(1.099557428756428, 0, -pi/2) q[8];
U(pi/2, -pi, 1.0995574287564276) q[9];
cz q[8], q[9];
U(pi/2, 2*pi/5, -pi) q[8];
U(pi, -pi, 0) q[9];
cz q[9], q[8];
U(pi/2, 0, pi) q[8];
U(pi, -pi/2, -pi) q[9];
cz q[8], q[9];
U(0, 0, pi/2) q[8];
U(pi/2, 0, pi/2) q[9];
cz q[8], q[9];
U(pi/2, 2*pi/5, -pi) q[8];
U(pi, -pi, 0) q[9];
cz q[9], q[8];
U(pi/2, 0, pi) q[8];
U(pi, -pi/2, -pi) q[9];
cz q[8], q[9];
U(2.084821383478164, 2.450625334071656, 1.3351338634352352) q[8];
cz q[7], q[8];
U(pi/2, 2*pi/5, -pi) q[7];
U(pi, -pi, 0) q[8];
cz q[8], q[7];
U(pi/2, 0, pi) q[7];
U(pi, -pi/2, -pi) q[8];
cz q[7], q[8];
U(1.099557428756428, 0, -pi/2) q[7];
U(pi/2, -pi, 1.0995574287564276) q[8];
cz q[7], q[8];
U(pi/2, 2*pi/5, -pi) q[7];
U(pi, -pi, 0) q[8];
cz q[8], q[7];
U(pi/2, 0, pi) q[7];
U(pi, -pi/2, -pi) q[8];
cz q[7], q[8];
U(0, 0, pi/2) q[7];
U(pi/2, 0, pi/2) q[8];
cz q[7], q[8];
U(pi/2, 2*pi/5, -pi) q[7];
U(pi, -pi, 0) q[8];
cz q[8], q[7];
U(pi/2, 0, pi) q[7];
U(pi, -pi/2, -pi) q[8];
cz q[7], q[8];
U(1.1546395978862831, 1.6188980331309324, 2.9147019705055026) q[7];
U(2.67079632678808, 1.0999999999931829, 2.6707963267880794) q[8];
U(1.0107821915673239, 1.8131315386304223, -0.6566734307263116) q[9];
U(2.588865794424887, -1.168104532357039, -0.7971524685638545) q[14];
U(1.7780257989479222, -2.6253978644933977, -2.7157522178460067) q[15];
cz q[14], q[15];
U(pi/2, 2*pi/5, -pi) q[14];
U(pi, -pi, 0) q[15];
cz q[15], q[14];
U(pi/2, 0, pi) q[14];
U(pi, -pi/2, -pi) q[15];
cz q[14], q[15];
U(1.099557428756428, 0, -pi/2) q[14];
U(pi/2, -pi, 1.0995574287564276) q[15];
cz q[14], q[15];
U(pi/2, 2*pi/5, -pi) q[14];
U(pi, -pi, 0) q[15];
cz q[15], q[14];
U(pi/2, 0, pi) q[14];
U(pi, -pi/2, -pi) q[15];
cz q[14], q[15];
U(0, 0, pi/2) q[14];
U(pi/2, 0, pi/2) q[15];
cz q[14], q[15];
U(pi/2, 2*pi/5, -pi) q[14];
U(pi, -pi, 0) q[15];
cz q[15], q[14];
U(pi/2, 0, pi) q[14];
U(pi, -pi/2, -pi) q[15];
cz q[14], q[15];
U(2.084821383478164, 2.450625334071656, 1.3351338634352352) q[14];
cz q[13], q[14];
U(pi/2, 2*pi/5, -pi) q[13];
U(pi, -pi, 0) q[14];
cz q[14], q[13];
U(pi/2, 0, pi) q[13];
U(pi, -pi/2, -pi) q[14];
cz q[13], q[14];
U(1.099557428756428, 0, -pi/2) q[13];
U(pi/2, -pi, 1.0995574287564276) q[14];
cz q[13], q[14];
U(pi/2, 2*pi/5, -pi) q[13];
U(pi, -pi, 0) q[14];
cz q[14], q[13];
U(pi/2, 0, pi) q[13];
U(pi, -pi/2, -pi) q[14];
cz q[13], q[14];
U(0, 0, pi/2) q[13];
U(pi/2, 0, pi/2) q[14];
cz q[13], q[14];
U(pi/2, 2*pi/5, -pi) q[13];
U(pi, -pi, 0) q[14];
cz q[14], q[13];
U(pi/2, 0, pi) q[13];
U(pi, -pi/2, -pi) q[14];
cz q[13], q[14];
U(1.5268029081266796, -1.2019829776562736, -1.7782294264593075) q[13];
cz q[12], q[13];
U(pi/2, pi/20, -pi) q[12];
U(pi, -pi, 0) q[13];
cz q[13], q[12];
U(pi/2, 0, pi) q[12];
U(pi, -pi/2, -pi) q[13];
cz q[12], q[13];
U(pi/20, -pi/2, pi/2) q[12];
U(pi/2, -3*pi/4, -4*pi/5) q[13];
cz q[12], q[13];
U(pi/2, pi/20, -pi) q[12];
U(pi, -pi, 0) q[13];
cz q[13], q[12];
U(pi/2, 0, pi) q[12];
U(pi, -pi/2, -pi) q[13];
cz q[12], q[13];
U(7*pi/10, -pi/4, -pi/2) q[12];
cz q[11], q[12];
U(pi/2, pi/20, -pi) q[11];
U(pi, -pi, 0) q[12];
cz q[12], q[11];
U(pi/2, 0, pi) q[11];
U(pi, -pi/2, -pi) q[12];
cz q[11], q[12];
U(pi/20, -pi/2, pi/2) q[11];
U(pi/2, -3*pi/4, -4*pi/5) q[12];
cz q[11], q[12];
U(pi/2, pi/20, -pi) q[11];
U(pi, -pi, 0) q[12];
cz q[12], q[11];
U(pi/2, 0, pi) q[11];
U(pi, -pi/2, -pi) q[12];
cz q[11], q[12];
U(1.3727085346750192, 2.129204585735234, 1.3593715260007677) q[11];
cz q[10], q[11];
U(pi/2, 2*pi/5, -pi) q[10];
U(pi, -pi, 0) q[11];
cz q[11], q[10];
U(pi/2, 0, pi) q[10];
U(pi, -pi/2, -pi) q[11];
cz q[10], q[11];
U(1.099557428756428, 0, -pi/2) q[10];
U(pi/2, -pi, 1.0995574287564276) q[11];
cz q[10], q[11];
U(pi/2, 2*pi/5, -pi) q[10];
U(pi, -pi, 0) q[11];
cz q[11], q[10];
U(pi/2, 0, pi) q[10];
U(pi, -pi/2, -pi) q[11];
cz q[10], q[11];
U(0, 0, pi/2) q[10];
U(pi/2, 0, pi/2) q[11];
cz q[10], q[11];
U(pi/2, 2*pi/5, -pi) q[10];
U(pi, -pi, 0) q[11];
cz q[11], q[10];
U(pi/2, 0, pi) q[10];
U(pi, -pi/2, -pi) q[11];
cz q[10], q[11];
U(2.084821383478164, 2.450625334071656, 1.3351338634352352) q[10];
cz q[9], q[10];
U(pi, -pi, 0) q[10];
U(1.0107821915673239, 1.8131315386304223, -0.6566734307263116) q[11];
U(0.5527268591649064, 1.9734881212327533, 1.4254709992818135) q[12];
U(pi/2, -pi, -1.7278759594743862) q[13];
U(1.1248260385203375, -2.985381781521503, 2.3207042664208153) q[14];
U(1.0107821915673239, 1.8131315386304223, -0.6566734307263116) q[15];
cz q[15], q[0];
U(pi, -pi, 0) q[0];
U(pi/2, 2*pi/5, -pi) q[15];
cz q[0], q[15];
U(pi, -pi/2, -pi) q[0];
U(pi/2, 0, pi) q[15];
cz q[15], q[0];
U(pi/2, -pi, 1.0995574287564276) q[0];
U(1.099557428756428, 0, -pi/2) q[15];
cz q[15], q[0];
U(pi, -pi, 0) q[0];
U(pi/2, 2*pi/5, -pi) q[15];
cz q[0], q[15];
U(pi, -pi/2, -pi) q[0];
U(pi/2, 0, pi) q[15];
cz q[15], q[0];
U(pi/2, 0, pi/2) q[0];
U(0, 0, pi/2) q[15];
cz q[15], q[0];
U(pi, -pi, 0) q[0];
U(pi/2, 2*pi/5, -pi) q[15];
cz q[0], q[15];
U(pi, -pi/2, -pi) q[0];
U(pi/2, 0, pi) q[15];
cz q[15], q[0];
U(1.1248260385203375, -2.985381781521503, 2.3207042664208153) q[0];
cz q[0], q[1];
U(pi/2, pi/20, -pi) q[0];
U(pi, -pi, 0) q[1];
cz q[1], q[0];
U(pi, -pi/2, -pi) q[1];
U(pi/2, 0, pi) q[0];
cz q[0], q[1];
U(pi/20, -pi/2, pi/2) q[0];
U(pi/2, -3*pi/4, -4*pi/5) q[1];
cz q[0], q[1];
U(pi/2, pi/20, -pi) q[0];
U(pi, -pi, 0) q[1];
cz q[1], q[0];
U(pi, -pi/2, -pi) q[1];
U(pi/2, 0, pi) q[0];
cz q[0], q[1];
U(7*pi/10, -pi/4, -pi/2) q[0];
U(pi/2, -pi, -1.7278759594743862) q[1];
cz q[1], q[2];
U(pi/2, pi/20, -pi) q[1];
U(1.5268029081266796, -1.2019829776562736, -1.7782294264593075) q[15];
cz q[14], q[15];
U(pi/2, pi/20, -pi) q[14];
U(pi, -pi, 0) q[15];
cz q[15], q[14];
U(pi/2, 0, pi) q[14];
U(pi, -pi/2, -pi) q[15];
cz q[14], q[15];
U(pi/20, -pi/2, pi/2) q[14];
U(pi/2, -3*pi/4, -4*pi/5) q[15];
cz q[14], q[15];
U(pi/2, pi/20, -pi) q[14];
U(pi, -pi, 0) q[15];
cz q[15], q[14];
U(pi/2, 0, pi) q[14];
U(pi, -pi/2, -pi) q[15];
cz q[14], q[15];
U(7*pi/10, -pi/4, -pi/2) q[14];
cz q[13], q[14];
U(pi/2, pi/20, -pi) q[13];
U(pi, -pi, 0) q[14];
cz q[14], q[13];
U(pi/2, 0, pi) q[13];
U(pi, -pi/2, -pi) q[14];
cz q[13], q[14];
U(pi/20, -pi/2, pi/2) q[13];
U(pi/2, -3*pi/4, -4*pi/5) q[14];
cz q[13], q[14];
U(pi/2, pi/20, -pi) q[13];
U(pi, -pi, 0) q[14];
cz q[14], q[13];
U(pi/2, 0, pi) q[13];
U(pi, -pi/2, -pi) q[14];
cz q[13], q[14];
U(1.3727085346750192, 2.129204585735234, 1.3593715260007677) q[13];
cz q[12], q[13];
U(pi/2, 2*pi/5, -pi) q[12];
U(pi, -pi, 0) q[13];
cz q[13], q[12];
U(pi/2, 0, pi) q[12];
U(pi, -pi/2, -pi) q[13];
cz q[12], q[13];
U(1.099557428756428, 0, -pi/2) q[12];
U(pi/2, -pi, 1.0995574287564276) q[13];
cz q[12], q[13];
U(pi/2, 2*pi/5, -pi) q[12];
U(pi, -pi, 0) q[13];
cz q[13], q[12];
U(pi/2, 0, pi) q[12];
U(pi, -pi/2, -pi) q[13];
cz q[12], q[13];
U(0, 0, pi/2) q[12];
U(pi/2, 0, pi/2) q[13];
cz q[12], q[13];
U(pi/2, 2*pi/5, -pi) q[12];
U(pi, -pi, 0) q[13];
cz q[13], q[12];
U(pi/2, 0, pi) q[12];
U(pi, -pi/2, -pi) q[13];
cz q[12], q[13];
U(2.084821383478164, 2.450625334071656, 1.3351338634352352) q[12];
cz q[11], q[12];
U(pi/2, 2*pi/5, -pi) q[11];
U(pi, -pi, 0) q[12];
cz q[12], q[11];
U(pi/2, 0, pi) q[11];
U(pi, -pi/2, -pi) q[12];
cz q[11], q[12];
U(1.099557428756428, 0, -pi/2) q[11];
U(pi/2, -pi, 1.0995574287564276) q[12];
cz q[11], q[12];
U(pi/2, 2*pi/5, -pi) q[11];
U(pi, -pi, 0) q[12];
cz q[12], q[11];
U(pi/2, 0, pi) q[11];
U(pi, -pi/2, -pi) q[12];
cz q[11], q[12];
U(0, 0, pi/2) q[11];
U(pi/2, 0, pi/2) q[12];
cz q[11], q[12];
U(pi/2, 2*pi/5, -pi) q[11];
U(pi, -pi, 0) q[12];
cz q[12], q[11];
U(pi/2, 0, pi) q[11];
U(pi, -pi/2, -pi) q[12];
cz q[11], q[12];
U(1.1546395978862831, 1.6188980331309324, 2.9147019705055026) q[11];
U(2.67079632678808, 1.0999999999931829, 2.6707963267880794) q[12];
U(1.0107821915673239, 1.8131315386304223, -0.6566734307263116) q[13];
U(0.5527268591649064, 1.9734881212327533, 1.4254709992818135) q[14];
U(pi/2, -pi, -1.7278759594743862) q[15];
cz q[15], q[0];
U(pi, -pi, 0) q[0];
U(pi/2, pi/20, -pi) q[15];
cz q[0], q[15];
U(pi, -pi/2, -pi) q[0];
U(pi/2, 0, pi) q[15];
cz q[15], q[0];
U(pi/2, -3*pi/4, -4*pi/5) q[0];
U(pi/20, -pi/2, pi/2) q[15];
cz q[15], q[0];
U(pi, -pi, 0) q[0];
U(pi/2, pi/20, -pi) q[15];
cz q[0], q[15];
U(pi, -pi/2, -pi) q[0];
U(pi/2, 0, pi) q[15];
cz q[15], q[0];
U(0.5527268591649064, 1.9734881212327533, 1.4254709992818135) q[0];
U(1.3727085346750192, 2.129204585735234, 1.3593715260007677) q[15];
cz q[14], q[15];
U(pi/2, 2*pi/5, -pi) q[14];
U(pi, -pi, 0) q[15];
cz q[15], q[14];
U(pi/2, 0, pi) q[14];
U(pi, -pi/2, -pi) q[15];
cz q[14], q[15];
U(1.099557428756428, 0, -pi/2) q[14];
U(pi/2, -pi, 1.0995574287564276) q[15];
cz q[14], q[15];
U(pi/2, 2*pi/5, -pi) q[14];
U(pi, -pi, 0) q[15];
cz q[15], q[14];
U(pi/2, 0, pi) q[14];
U(pi, -pi/2, -pi) q[15];
cz q[14], q[15];
U(0, 0, pi/2) q[14];
U(pi/2, 0, pi/2) q[15];
cz q[14], q[15];
U(pi/2, 2*pi/5, -pi) q[14];
U(pi, -pi, 0) q[15];
cz q[15], q[14];
U(pi/2, 0, pi) q[14];
U(pi, -pi/2, -pi) q[15];
cz q[14], q[15];
U(2.084821383478164, 2.450625334071656, 1.3351338634352352) q[14];
cz q[13], q[14];
U(pi/2, 2*pi/5, -pi) q[13];
U(pi, -pi, 0) q[14];
cz q[14], q[13];
U(pi/2, 0, pi) q[13];
U(pi, -pi/2, -pi) q[14];
cz q[13], q[14];
U(1.099557428756428, 0, -pi/2) q[13];
U(pi/2, -pi, 1.0995574287564276) q[14];
cz q[13], q[14];
U(pi/2, 2*pi/5, -pi) q[13];
U(pi, -pi, 0) q[14];
cz q[14], q[13];
U(pi/2, 0, pi) q[13];
U(pi, -pi/2, -pi) q[14];
cz q[13], q[14];
U(0, 0, pi/2) q[13];
U(pi/2, 0, pi/2) q[14];
cz q[13], q[14];
U(pi/2, 2*pi/5, -pi) q[13];
U(pi, -pi, 0) q[14];
cz q[14], q[13];
U(pi/2, 0, pi) q[13];
U(pi, -pi/2, -pi) q[14];
cz q[13], q[14];
U(1.1546395978862831, 1.6188980331309324, 2.9147019705055026) q[13];
U(2.67079632678808, 1.0999999999931829, 2.6707963267880794) q[14];
U(1.0107821915673239, 1.8131315386304223, -0.6566734307263116) q[15];
U(pi, -pi, 0) q[2];
cz q[2], q[1];
U(pi/2, 0, pi) q[1];
U(pi, -pi/2, -pi) q[2];
cz q[1], q[2];
U(pi/20, -pi/2, pi/2) q[1];
U(pi/2, -3*pi/4, -4*pi/5) q[2];
cz q[1], q[2];
U(pi/2, pi/20, -pi) q[1];
U(pi, -pi, 0) q[2];
cz q[2], q[1];
U(pi/2, 0, pi) q[1];
U(pi, -pi/2, -pi) q[2];
cz q[1], q[2];
U(1.3727085346750192, 2.129204585735234, 1.3593715260007677) q[1];
cz q[0], q[1];
U(pi/2, 2*pi/5, -pi) q[0];
U(pi, -pi, 0) q[1];
cz q[1], q[0];
U(pi, -pi/2, -pi) q[1];
U(pi/2, 0, pi) q[0];
cz q[0], q[1];
U(1.099557428756428, 0, -pi/2) q[0];
U(pi/2, -pi, 1.0995574287564276) q[1];
cz q[0], q[1];
U(pi/2, 2*pi/5, -pi) q[0];
U(pi, -pi, 0) q[1];
cz q[1], q[0];
U(pi, -pi/2, -pi) q[1];
U(pi/2, 0, pi) q[0];
cz q[0], q[1];
U(0, 0, pi/2) q[0];
U(pi/2, 0, pi/2) q[1];
cz q[0], q[1];
U(pi/2, 2*pi/5, -pi) q[0];
U(pi, -pi, 0) q[1];
cz q[1], q[0];
U(pi, -pi/2, -pi) q[1];
U(pi/2, 0, pi) q[0];
cz q[0], q[1];
U(2.084821383478164, 2.450625334071656, 1.3351338634352352) q[0];
cz q[15], q[0];
U(pi, -pi, 0) q[0];
U(1.0107821915673239, 1.8131315386304223, -0.6566734307263116) q[1];
U(pi/2, 2*pi/5, -pi) q[15];
cz q[0], q[15];
U(pi, -pi/2, -pi) q[0];
U(pi/2, 0, pi) q[15];
cz q[15], q[0];
U(pi/2, -pi, 1.0995574287564276) q[0];
U(1.099557428756428, 0, -pi/2) q[15];
cz q[15], q[0];
U(pi, -pi, 0) q[0];
U(pi/2, 2*pi/5, -pi) q[15];
cz q[0], q[15];
U(pi, -pi/2, -pi) q[0];
U(pi/2, 0, pi) q[15];
cz q[15], q[0];
U(pi/2, 0, pi/2) q[0];
U(0, 0, pi/2) q[15];
cz q[15], q[0];
U(pi, -pi, 0) q[0];
U(pi/2, 2*pi/5, -pi) q[15];
cz q[0], q[15];
U(pi, -pi/2, -pi) q[0];
U(pi/2, 0, pi) q[15];
cz q[15], q[0];
U(2.67079632678808, 1.0999999999931829, 2.6707963267880794) q[0];
U(1.1546395978862831, 1.6188980331309324, 2.9147019705055026) q[15];
U(0.5527268591649064, 1.9734881212327533, 1.4254709992818135) q[2];
cz q[2], q[3];
U(pi/2, 2*pi/5, -pi) q[2];
U(pi, -pi, 0) q[3];
cz q[3], q[2];
U(pi/2, 0, pi) q[2];
U(pi, -pi/2, -pi) q[3];
cz q[2], q[3];
U(1.099557428756428, 0, -pi/2) q[2];
U(pi/2, -pi, 1.0995574287564276) q[3];
cz q[2], q[3];
U(pi/2, 2*pi/5, -pi) q[2];
U(pi, -pi, 0) q[3];
cz q[3], q[2];
U(pi/2, 0, pi) q[2];
U(pi, -pi/2, -pi) q[3];
cz q[2], q[3];
U(0, 0, pi/2) q[2];
U(pi/2, 0, pi/2) q[3];
cz q[2], q[3];
U(pi/2, 2*pi/5, -pi) q[2];
U(pi, -pi, 0) q[3];
cz q[3], q[2];
U(pi/2, 0, pi) q[2];
U(pi, -pi/2, -pi) q[3];
cz q[2], q[3];
U(2.084821383478164, 2.450625334071656, 1.3351338634352352) q[2];
cz q[1], q[2];
U(pi/2, 2*pi/5, -pi) q[1];
U(pi, -pi, 0) q[2];
cz q[2], q[1];
U(pi/2, 0, pi) q[1];
U(pi, -pi/2, -pi) q[2];
cz q[1], q[2];
U(1.099557428756428, 0, -pi/2) q[1];
U(pi/2, -pi, 1.0995574287564276) q[2];
cz q[1], q[2];
U(pi/2, 2*pi/5, -pi) q[1];
U(pi, -pi, 0) q[2];
cz q[2], q[1];
U(pi/2, 0, pi) q[1];
U(pi, -pi/2, -pi) q[2];
cz q[1], q[2];
U(0, 0, pi/2) q[1];
U(pi/2, 0, pi/2) q[2];
cz q[1], q[2];
U(pi/2, 2*pi/5, -pi) q[1];
U(pi, -pi, 0) q[2];
cz q[2], q[1];
U(pi/2, 0, pi) q[1];
U(pi, -pi/2, -pi) q[2];
cz q[1], q[2];
U(1.1546395978862831, 1.6188980331309324, 2.9147019705055026) q[1];
U(2.67079632678808, 1.0999999999931829, 2.6707963267880794) q[2];
U(1.0107821915673239, 1.8131315386304223, -0.6566734307263116) q[3];
cz q[3], q[4];
U(pi/2, 2*pi/5, -pi) q[3];
U(pi, -pi, 0) q[4];
cz q[4], q[3];
U(pi/2, 0, pi) q[3];
U(pi, -pi/2, -pi) q[4];
cz q[3], q[4];
U(1.099557428756428, 0, -pi/2) q[3];
U(pi/2, -pi, 1.0995574287564276) q[4];
cz q[3], q[4];
U(pi/2, 2*pi/5, -pi) q[3];
U(pi, -pi, 0) q[4];
cz q[4], q[3];
U(pi/2, 0, pi) q[3];
U(pi, -pi/2, -pi) q[4];
cz q[3], q[4];
U(0, 0, pi/2) q[3];
U(pi/2, 0, pi/2) q[4];
cz q[3], q[4];
U(pi/2, 2*pi/5, -pi) q[3];
U(pi, -pi, 0) q[4];
cz q[4], q[3];
U(pi/2, 0, pi) q[3];
U(pi, -pi/2, -pi) q[4];
cz q[3], q[4];
U(1.1546395978862831, 1.6188980331309324, 2.9147019705055026) q[3];
U(2.67079632678808, 1.0999999999931829, 2.6707963267880794) q[4];
U(pi/2, 2*pi/5, -pi) q[9];
cz q[10], q[9];
U(pi/2, 0, pi) q[9];
U(pi, -pi/2, -pi) q[10];
cz q[9], q[10];
U(pi/2, -pi, 1.0995574287564276) q[10];
U(1.099557428756428, 0, -pi/2) q[9];
cz q[9], q[10];
U(pi, -pi, 0) q[10];
U(pi/2, 2*pi/5, -pi) q[9];
cz q[10], q[9];
U(pi/2, 0, pi) q[9];
U(pi, -pi/2, -pi) q[10];
cz q[9], q[10];
U(pi/2, 0, pi/2) q[10];
U(0, 0, pi/2) q[9];
cz q[9], q[10];
U(pi, -pi, 0) q[10];
U(pi/2, 2*pi/5, -pi) q[9];
cz q[10], q[9];
U(pi/2, 0, pi) q[9];
U(pi, -pi/2, -pi) q[10];
cz q[9], q[10];
U(2.67079632678808, 1.0999999999931829, 2.6707963267880794) q[10];
U(1.1546395978862831, 1.6188980331309324, 2.9147019705055026) q[9];
ans[0] = measure q[0];
ans[1] = measure q[1];
ans[2] = measure q[2];
ans[3] = measure q[3];
ans[4] = measure q[4];
ans[5] = measure q[5];
ans[6] = measure q[6];
ans[7] = measure q[7];
ans[8] = measure q[8];
ans[9] = measure q[9];
ans[10] = measure q[10];
ans[11] = measure q[11];
ans[12] = measure q[12];
ans[13] = measure q[13];
ans[14] = measure q[14];
ans[15] = measure q[15];
