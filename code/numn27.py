from assimulo.ode import Explicit_Problem, Explicit_ODE_Exception, ID_PY_OK
from assimulo.explicit_ode import Explicit_ODE
import numpy as np
import scipy.optimize as opt
import matplotlib.pyplot as plt


class EntropyProblem(Explicit_Problem):
    def __init__(self, y0, func, eta, etaprim):
        self.t0 = 0
        self.y0 = y0
        self.func = func
        self.eta = eta
        self.etaprim = etaprim

    def rhs(self, t, y):
        return self.func(t, y)


class RK(Explicit_ODE):
    maxsteps = 10000

    def __init__(self, problem):
        Explicit_ODE.__init__(self, problem)
        self.b = self.get_b()
        self.c = self.get_c()
        self.A = self.get_A()

        self.options["h"] = 0.01
        self.statistics["nsteps"] = 0
        self.statistics["nfcns"] = 0

    def _get_h(self):
        return self.options["h"]

    def _set_h(self, new_h):
        self.options["h"] = float(new_h)

    def get_local_errors(self, tf, true_func):
        h = self._get_h()
        t = self.problem.t0
        e_list = [0]
        true_eval = self.problem.y0.copy()
        for i in range(self.maxsteps):
            t, y = self.step(t, true_eval, h)
            self.statistics["nsteps"] += 1
            if t > tf:
                break
            true_eval = true_func(t)
            e_list.append(true_eval - y)
        else:
            raise Explicit_ODE_Exception('Final time not reached within maximum number of steps')
        return e_list

    def integrate(self, t, y, tf, opts, error=False):
        h = self._get_h()
        y_list = [y]
        t_list = [t]
        for i in range(self.maxsteps):
            self.statistics["nsteps"] += 1
            if t + h >= tf:
                t, y = self.step(t_list[-1], y_list[-1], tf - t_list[-1])
                t_list.append(t)
                y_list.append(y)
                break
            else:
                t, y = self.step(t_list[-1], y_list[-1], h)
                t_list.append(t)
                y_list.append(y)
        else:
            raise Explicit_ODE_Exception('Final time not reached within maximum number of steps')

        return ID_PY_OK, t_list, y_list

    def step(self, t, y, h):
        Y = self.get_stage_values(t, y, h)
        f_vec = self.get_evals(t, h, Y)
        sumbf = np.zeros(np.shape(y))
        for i in range(len(f_vec)):
            sumbf += self.b[i] * f_vec[i]
        y_new = y + h * sumbf
        t_new = t + h
        return t_new, y_new

    def get_stage_values(self, t, y, h):
        Y = []
        for i in range(len(self.b)):
            temp_Y = y.copy()
            for j in range(len(Y)):
                self.statistics["nfcns"] += 1
                temp_Y += h * self.A[i, j] * self.problem.rhs(t + self.c[j], Y[j])
            Y.append(temp_Y)
        return np.array(Y)

    def get_evals(self, t, step_size, stage_values):
        self.statistics["nfcns"] += len(self.c)
        return np.array([self.problem.rhs(t + self.c[i] * step_size, stage_values[i]) for i in range(len(self.c))])


class IDT(RK):
    maxsteps = 100000

    def __init__(self, problem):
        RK.__init__(self, problem)
        self.options["h"] = 0.01
        self.statistics["nsteps"] = 0
        self.statistics["nfcns"] = 0

    def _get_h(self):
        return self.options["h"]

    def _set_h(self, new_h):
        self.options["h"] = float(new_h)

    def integrate(self, t, y, tf, opts):
        h = self._get_h()
        y_list = [y]
        t_list = [t]
        for i in range(self.maxsteps):
            t, y = self.step(t, y, h)
            self.statistics["nsteps"] += 1
            if t >= tf:
                t, y = RK.step(self, t_list[-1], y_list[-1], tf - t_list[-1])
                t_list.append(t)
                y_list.append(y)
                break
            t_list.append(t)
            y_list.append(y)
        else:
            raise Explicit_ODE_Exception('Final time not reached within maximum number of steps')
        return ID_PY_OK, t_list, y_list

    def step(self, t, y, h):
        Y = self.get_stage_values(t, y, h)
        f_vec = self.get_evals(t, h, Y)
        sumbf = np.zeros(np.shape(y))
        for i in range(len(f_vec)):
            sumbf += self.b[i] * f_vec[i]
        r_func = self.get_r_func(y, Y, f_vec, h)
        gamma = opt.root_scalar(r_func, bracket=(0.5, 2)).root
        y_new = y + h * gamma * sumbf
        t_new = t + h
        return t_new, y_new

    def get_r_func(self, y, stage_values, f_eval, h):
        b = self.b
        sumbf, temp = np.zeros(np.shape(y)), 0
        for i in range(len(f_eval)):
            sumbf += b[i] * f_eval[i]
            temp += b[i] * np.inner(self.problem.etaprim(stage_values[i]), f_eval[i])
        return lambda gamma: self.problem.eta(y + gamma * h * sumbf) - self.problem.eta(y) - gamma * h * temp

    def r_plot(self, filename="r_of_gamma.pdf"):
        y = self.problem.y0
        t = self.problem.t0
        for h in [0.12, 0.10, 0.08]:
            Y = self.get_stage_values(t, y, h)
            f_vec = self.get_evals(t, h, Y)
            sumbf = np.zeros(np.shape(y))
            for i in range(len(f_vec)):
                sumbf += self.b[i] * f_vec[i]
            r_func = self.get_r_func(y, Y, f_vec, h)
            gamma_list = np.linspace(-0.1, 1.2, 100)
            r_list = [r_func(g) for g in gamma_list]
            plt.plot(gamma_list, r_list, ".", label=f"$\\Delta t = ${h}")
        plt.xlabel("$\\gamma$")
        plt.ylabel("$r(\\gamma)$")
        plt.legend()
        plt.grid()
        plt.savefig(filename)
        plt.show()

        return gamma_list, r_list

    def r_plot2(self):
        h_list = np.logspace(-3, 0, 100)
        y = self.problem.y0
        t = self.problem.t0
        r_list = []
        for h in h_list:
            Y = self.get_stage_values(t, y, h)
            f_vec = self.get_evals(t, h, Y)
            sumbf = np.zeros(np.shape(y))
            for i in range(len(f_vec)):
                sumbf += self.b[i] * f_vec[i]
            r_func = self.get_r_func(y, Y, f_vec, h)
            r_list.append(np.abs(r_func(1)))
        # plt.loglog(h_list, r_list, ".")
        # plt.show()
        return h_list, r_list


class RRK(IDT):
    maxsteps = 100000

    def __init__(self, problem):
        IDT.__init__(self, problem)
        self.options["h"] = 0.01
        self.statistics["nsteps"] = 0
        self.statistics["nfcns"] = 0

    def step(self, t, y, h):
        Y = self.get_stage_values(t, y, h)
        f_vec = self.get_evals(t, h, Y)
        sumbf = np.zeros(np.shape(y))
        for i in range(len(f_vec)):
            sumbf += self.b[i] * f_vec[i]
        r_func = self.get_r_func(y, Y, f_vec, h)
        gamma = opt.root_scalar(r_func, bracket=(0.5, 2)).root
        y_new = y + h * gamma * sumbf
        t_new = t + h * gamma
        return t_new, y_new

    def integrate(self, t, y, tf, opts):
        h = self._get_h()
        y_list = [y]
        t_list = [t]
        for i in range(self.maxsteps):
            t, y = self.step(t, y, h)
            self.statistics["nsteps"] += 1
            if t >= tf:
                t, y = RK.step(self, t_list[-1], y_list[-1], tf - t_list[-1])
                t_list.append(t)
                y_list.append(y)
                break
            t_list.append(t)
            y_list.append(y)
        else:
            raise Explicit_ODE_Exception('Final time not reached within maximum number of steps')
        return ID_PY_OK, t_list, y_list


class classicRK4(RK):
    name = "Classic RK4"

    def get_b(self):
        return np.array([1, 2, 2, 1]) / 6

    def get_c(self):
        return np.array([0, 0.5, 0.5, 1])

    def get_A(self):
        return np.array([[0, 0, 0, 0],
                         [0.5, 0, 0, 0],
                         [0, 0.5, 0, 0],
                         [0, 0, 1, 0]])


class classicSSPRK33(RK):
    name = "Classic SSPRK(3,3)"

    def get_b(self):
        return np.array([1, 1, 4]) / 6

    def get_c(self):
        return np.array([0, 1, 0.5])

    def get_A(self):
        return np.array([[0, 0, 0],
                         [1, 0, 0],
                         [0.25, 0.25, 0]])


class classicVRK96(RK):
    name = "Classic VRK(9,6)"

    def get_b(self):
        return np.array([.7638888888888888888888888888888888888889e-1,
                         0.,
                         0.,
                         .3694083694083694083694083694083694083694,
                         0.,
                         .2480158730158730158730158730158730158730,
                         .2367424242424242424242424242424242424242,
                         .6944444444444444444444444444444444444444e-1,
                         0.])

    def get_c(self):
        return np.array([0.,
                         .18,
                         .1666666666666666666666666666666666666667,
                         .25,
                         .53,
                         .6,
                         .8,
                         1.,
                         1.])

    def get_A(self):
        a = np.zeros((9, 9))
        a[1, 0] = .18

        a[2, 0:2] = [.8950617283950617283950617283950617283951e-1,
                     .7716049382716049382716049382716049382716e-1]

        a[3, 0:3] = [.625e-1,
                     0.,
                     .1875]

        a[4, 0:4] = [.316516,
                     0.,
                     -1.044948,
                     1.258432]

        a[5, 0:5] = [.2723261273648562625722506556667430550251,
                     0.,
                     -.8251336032388663967611336032388663967611,
                     1.048091767881241565452091767881241565452,
                     .1047157079927685687367911796908817762840]

        a[6, 0:6] = [-.1669941859971651431432960727896179733320,
                     0.,
                     .6317085020242914979757085020242914979757,
                     .1746104455277387608214675883848816179643,
                     -1.066535645908606612252519473401868067778,
                     1.227210884353741496598639455782312925170]

        a[7, 0:7] = [.3642375168690958164642375168690958164642,
                     0.,
                     -.2040485829959514170040485829959514170040,
                     -.3488373781606864313631230924464007170774,
                     3.261932303285686744333360874714258172905,
                     -2.755102040816326530612244897959183673469,
                     .6818181818181818181818181818181818181818]

        a[8, 0:8] = [.7638888888888888888888888888888888888889e-1,
                     0.,
                     0.,
                     .3694083694083694083694083694083694083694,
                     0.,
                     .2480158730158730158730158730158730158730,
                     .2367424242424242424242424242424242424242,
                     .6944444444444444444444444444444444444444e-1]
        return a


class classicVRK138(RK):
    name = "Classic VRK(13,8)"

    def get_b(self):
        return np.array([.4472956466669571420301584042904938246647e-1,
                         0.,
                         0.,
                         0.,
                         0.,
                         .1569103352770819981336869801072664540918,
                         .1846097340815163774070245187352627789204,
                         .2251638060208699104247941940035072197092,
                         .1479461565197023468700517988544914175374,
                         .7605554244495582526979836191033649101273e-1,
                         .1227729023501861961082434631592143738854,
                         .4181195863899163158338484280087188237679e-1,
                         0.])

    def get_c(self):
        return np.array([0.,
                         .25,
                         .1128884514435695538057742782152230971129,
                         .1693326771653543307086614173228346456693,
                         .424,
                         .509,
                         .867,
                         .15,
                         .7090680365138684008060140010282474786750,
                         .32,
                         .45,
                         1.,
                         1.])

    def get_A(self):
        a = np.zeros((13, 13))
        a[1, 0] = .25

        a[2, 0:2] = [.8740084650491523205268632759487741197705e-1,
                     .2548760493865432175308795062034568513581e-1]

        a[3, 0:3] = [.4233316929133858267716535433070866141732e-1,
                     0.,
                     .1269995078740157480314960629921259842520]

        a[4, 0:4] = [.4260950588874226149488144523757227409094,
                     0.,
                     -1.598795284659152326542773323065718111709,
                     1.596700225771729711593958870689995370799]

        a[5, 0:5] = [.5071933729671392951509061813851363923933e-1,
                     0.,
                     0.,
                     .2543337726460040758275471440887777803137,
                     .2039468900572819946573622377727085804470]

        a[6, 0:6] = [-.2900037471752311097038837928542589612409,
                     0.,
                     0.,
                     1.344187391026078988943868110941433700318,
                     -2.864777943361442730961110382703656282947,
                     2.677594299510594851721126064616481543870]

        a[7, 0:7] = [.9853501133799354646974040298072701428476e-1,
                     0.,
                     0.,
                     0.,
                     .2219268063075138484202403649819738790358,
                     -.1814062291180699431269033828807395245747,
                     .1094441147256254823692261491803863125415e-1]

        a[8, 0:8] = [.3871105254573114467944461816516637340565,
                     0.,
                     0.,
                     -1.442445497485527757125674555307792776717,
                     2.905398189069950931769134644923384844174,
                     -1.853771069630105929084333267581197802518,
                     .1400364809872815426949732510977124147922,
                     .5727394081149581657574677462444770648875]

        a[9, 0:9] = [-.1612440344443930810063001619791348059544,
                     0.,
                     0.,
                     -.1733960295735898408357840447396256789490,
                     -1.301289281406514740601681274517249252974,
                     1.137950375173861730855879213143100347212,
                     -.3174764966396688010692352113804302469898e-1,
                     .9335129382493366643981106448605688485659,
                     -.8378631833473385270330085562961643320150e-1]

        a[10, 0:10] = [-.1919944488158953328151080465148357607314e-1,
                       0.,
                       0.,
                       .2733085726526428490794232625401612427562,
                       -.6753497320694437291969161121094238085624,
                       .3415184981384601607173848997472838271198,
                       -.6795006480337577247892051619852462939191e-1,
                       .9659175224762387888426558649121637650975e-1,
                       .1325308251118210118072103846654538995123,
                       .3685495936038611344690632995153166681295]

        a[11, 0:11] = [.6091877403645289867688841211158881778458,
                       0.,
                       0.,
                       -2.272569085898001676899980093141308839972,
                       4.757898342694029006815525588191478549755,
                       -5.516106706692758482429468966784424824484,
                       .2900596369680119270909581856594617437818,
                       .5691423963359036822910985845480184914563,
                       .7926795760332167027133991620589332757995,
                       .1547372045328882289412619077184989823205,
                       1.614970895662181624708321510633454443497]

        a[12, 0:12] = [.8873576220853471966321169405198102270488,
                       0.,
                       0.,
                       -2.975459782108536755851363280470930158198,
                       5.600717009488163059799039254835009892383,
                       -5.915607450536674468001493018994165735184,
                       .2202968915613492701687914254080763833124,
                       .1015509782446221666614327134090299699755,
                       1.151434564738605590978039775212585055356,
                       1.929710166527123939613436190080584365307,
                       0.,
                       0.]
        return a


class relaxedRK4(RRK):
    name = "Relaxed RK4"

    def get_b(self):
        return classicRK4.get_b(self)

    def get_c(self):
        return classicRK4.get_c(self)

    def get_A(self):
        return classicRK4.get_A(self)


class relaxedSSPRK33(RRK):
    name = "Relaxed SSPRK(3,3)"

    def get_b(self):
        return classicSSPRK33.get_b(self)

    def get_c(self):
        return classicSSPRK33.get_c(self)

    def get_A(self):
        return classicSSPRK33.get_A(self)


class relaxedVRK96(RRK):
    name = "Relaxed VRK(9,6)"

    def get_b(self):
        return classicVRK96.get_b(self)

    def get_c(self):
        return classicVRK96.get_c(self)

    def get_A(self):
        return classicVRK96.get_A(self)


class relaxedVRK138(RRK):
    name = "Relaxed VRK(13,8)"

    def get_b(self):
        return classicVRK138.get_b(self)

    def get_c(self):
        return classicVRK138.get_c(self)

    def get_A(self):
        return classicVRK138.get_A(self)


class IDT_RK4(IDT):
    name = "IDT RK4"

    def get_b(self):
        return classicRK4.get_b(self)

    def get_c(self):
        return classicRK4.get_c(self)

    def get_A(self):
        return classicRK4.get_A(self)


class IDT_SSPRK33(IDT):
    name = "IDT SSPRK(3,3)"

    def get_b(self):
        return classicSSPRK33.get_b(self)

    def get_c(self):
        return classicSSPRK33.get_c(self)

    def get_A(self):
        return classicSSPRK33.get_A(self)


class IDT_VRK96(IDT):
    name = "IDT VRK(9,6)"

    def get_b(self):
        return classicVRK96.get_b(self)

    def get_c(self):
        return classicVRK96.get_c(self)

    def get_A(self):
        return classicVRK96.get_A(self)


class IDT_VRK138(IDT):
    name = "IDT VRK(13,8)"

    def get_b(self):
        return classicVRK138.get_b(self)

    def get_c(self):
        return classicVRK138.get_c(self)

    def get_A(self):
        return classicVRK138.get_A(self)


if __name__ == "__main__":

    def dotheplot(solver_list, problem, exact_sol, end_point, titles, filename=None):
        if filename is None:
            filename = f"{titles[1]}.pdf"
        h_list = np.logspace(-3, 0, 100)
        plot_vals = []
        for solver in solver_list:
            temp_plot_vals = []
            for h in h_list:
                approx = solver(problem)
                approx._set_h(h)
                t_list, y_list = approx.simulate(end_point)
                temp_plot_vals.append(np.linalg.norm(exact_sol(t_list[-2]) - y_list[-2]))
            plot_vals.append(temp_plot_vals)

        coeffs = [None, 1, 0.5, 0.05, 0.005, 0.0005, 0.00005]

        for i in range(len(plot_vals)):
            plt.loglog(h_list, plot_vals[i], ".", label=titles[i])

        plot_h = [h_list[int(len(h_list) / 4)], h_list[int(len(h_list) / 2)]]
        plt.loglog(plot_h, [coeffs[2] * h**2 for h in plot_h],
                   label="$\\mathcal{O}(\\Delta t^2)$", color="gray", marker="^")
        plt.loglog(plot_h, [coeffs[3] * h**3 for h in plot_h],
                   label="$\\mathcal{O}(\\Delta t^3)$", color="gray", marker="s")
        plt.loglog(plot_h, [coeffs[4] * h**4 for h in plot_h],
                   label="$\\mathcal{O}(\\Delta t^4)$", color="gray", marker="o")
        plot_h = [h_list[int(3 * len(h_list) / 8)], h_list[int(5 * len(h_list) / 8)]]
        plt.loglog(plot_h, [coeffs[5] * h**5 for h in plot_h],
                   label="$\\mathcal{O}(\\Delta t^5)$", color="gray", marker=">")
        plot_h = [h_list[int(len(h_list) / 2)], h_list[int(3 * len(h_list) / 4)]]
        plt.loglog(plot_h, [coeffs[6] * h**6 for h in plot_h],
                   label="$\\mathcal{O}(\\Delta t^6)$", color="gray", marker="<")

        plt.ylim((1e-15, 1.5))
        plt.xlim((0.004, 1.1))
        plt.xlabel("$\\Delta t$")
        plt.ylabel("Error at $t \\approx 5$")
        plt.legend()
        plt.grid()
        plt.savefig(filename)
        plt.show()

    def do_problem_1():
        solvers_classic = [classicSSPRK33, classicRK4]
        solvers_RRK = [relaxedSSPRK33, relaxedRK4]
        solvers_IDT = [IDT_SSPRK33, IDT_RK4]

        def func1(t, y):
            return np.array([-np.exp(y[1]), np.exp(y[0])])

        def entropy1(y):
            return np.exp(y[0]) + np.exp(y[1])

        def entropyprime1(y):
            return np.array([np.exp(y[0]), np.exp(y[1])])
        test_problem1 = EntropyProblem(np.array([1, 0.5]), func1, entropy1, entropyprime1)

        def do_plot_1a():
            solver = relaxedSSPRK33(test_problem1)
            solver.r_plot("P1_r_of_gamma.pdf")

        def do_plot_1b():
            for solver in solvers_RRK:
                approx = solver(test_problem1)
                h_list, r_list = approx.r_plot2()
                plt.loglog(h_list, r_list, ".", label=approx.name)
            coeffs = [None, 1, 0.5, 0.05, 0.005, 0.0005, 0.00005]
            plot_h = [h_list[int(len(h_list) / 4)], h_list[int(len(h_list) / 2)]]
            plt.loglog(plot_h, [coeffs[2] * h**2 for h in plot_h],
                       label="$\\mathcal{O}(\\Delta t^2)$", color="gray", marker="^")
            plt.loglog(plot_h, [coeffs[3] * h**3 for h in plot_h],
                       label="$\\mathcal{O}(\\Delta t^3)$", color="gray", marker="s")
            plt.loglog(plot_h, [coeffs[4] * h**4 for h in plot_h],
                       label="$\\mathcal{O}(\\Delta t^4)$", color="gray", marker="o")
            plot_h = [h_list[int(3 * len(h_list) / 8)], h_list[int(5 * len(h_list) / 8)]]
            plt.loglog(plot_h, [coeffs[5] * h**5 for h in plot_h],
                       label="$\\mathcal{O}(\\Delta t^5)$", color="gray", marker=">")
            plot_h = [h_list[int(len(h_list) / 2)], h_list[int(3 * len(h_list) / 4)]]
            plt.loglog(plot_h, [coeffs[6] * h**6 for h in plot_h],
                       label="$\\mathcal{O}(\\Delta t^6)$", color="gray", marker="<")
            plt.xlabel("$\\Delta t$")
            plt.ylabel("$|r(\\gamma = 1)|$")
            plt.legend()
            plt.grid()
            plt.savefig("P1_r_at_1.pdf")
            plt.show()

        def exact_sol(t):
            sqepe = np.sqrt(np.e) + np.e
            exp_sqepet = np.exp(sqepe * t)
            u1_nom = np.log(np.e + np.exp(3 / 2))
            u2_nom = np.log(exp_sqepet * sqepe)
            denom = np.log(np.sqrt(np.e) + exp_sqepet)
            return np.array([u1_nom - denom, u2_nom - denom])

        end_point = 5

        def do_plot_2a():
            dotheplot(solvers_classic, test_problem1, exact_sol, end_point,
                      titles=["Regular SSPRK(3,3)", "Regular RK(4,4)", "Regular VRK(9,6)"],
                      filename="P1_Regular_RK4.pdf")

        def do_plot_2b():
            dotheplot(solvers_RRK, test_problem1, exact_sol, end_point,
                      titles=["Relaxed SSPRK(3,3)", "Relaxed RK(4,4)", "Relaxed VRK(9,6)"],
                      filename="P1_Relaxed_RK4.pdf")

        def do_plot_2c():
            dotheplot(solvers_IDT, test_problem1, exact_sol, end_point,
                      titles=["IDT SSPRK(3,3)", "IDT RK(4,4)", "IDT VRK(9,6)"],
                      filename="P1_IDT_RK4.pdf")

        def do_gamma_plot():
            RKtypes = ["SSPRK(3,3)", "RK(4,4)"]
            RKfilenames = ["SSPRK33", "RK44"]
            stepsize = 0.01
            end_point = 5
            for i in range(len(solvers_classic)):
                approx = solvers_classic[i](test_problem1)
                approx._set_h(stepsize)
                t_list, y_list = approx.simulate(end_point)
                plt.semilogy(t_list[:-1], [np.abs(entropy1(y_list[j]) - entropy1(exact_sol(t_list[j]))) for j in range(len(t_list[:-1]))], label="Classic method")

                approx = solvers_RRK[i](test_problem1)
                approx._set_h(stepsize)
                t_list, y_list = approx.simulate(end_point)
                plt.semilogy(t_list[:-1], [np.abs(entropy1(y_list[j]) - entropy1(exact_sol(t_list[j]))) for j in range(len(t_list[:-1]))], label="RRK")

                approx = solvers_IDT[i](test_problem1)
                approx._set_h(stepsize)
                t_list, y_list = approx.simulate(end_point)
                plt.semilogy(t_list[:-1], [np.abs(entropy1(y_list[j]) - entropy1(exact_sol(t_list[j]))) for j in range(len(t_list[:-1]))], label="IDT")

                plt.title(f"Error in entropy of approximations\n of problem 1 using {RKtypes[i]} with $\\Delta t= {stepsize}$")
                plt.xlabel("Time, $t$")
                plt.ylabel("$|\\eta(u_i)-\\eta (u(t_i))|$")
                plt.legend()
                plt.grid()
                plt.savefig(f"P1_eta_error_{RKfilenames[i]}.pdf")
                plt.show()

        # do_plot_1a()
        # do_plot_1b()
        # do_plot_2a()
        # do_plot_2b()
        # do_plot_2c()
        do_gamma_plot()

    def do_problem_2():
        solvers_classic = [classicSSPRK33, classicRK4, classicVRK96, classicVRK138]
        solvers_RRK = [relaxedSSPRK33, relaxedRK4, relaxedVRK96, relaxedVRK138]
        solvers_IDT = [IDT_SSPRK33, IDT_RK4, IDT_VRK96, IDT_VRK138]

        def func2(t, y):
            return np.array([-np.exp(y[0])])

        def entropy2(y):
            return np.exp(y[0])

        def entropyprime2(y):
            return np.array([np.exp(y[0])])

        test_problem2 = EntropyProblem(np.array([0.5]), func2, entropy2, entropyprime2)

        def exact_sol(t):
            return np.array([-np.log(np.exp(-0.5) + t)])

        def do_plot_1a_2():
            solver = relaxedSSPRK33(test_problem2)
            solver.r_plot("P2_r_of_gamma.pdf")

        def do_plot_1b_2():
            for solver in solvers_RRK:
                approx = solver(test_problem2)
                h_list, r_list = approx.r_plot2()
                plt.loglog(h_list, r_list, ".", label=approx.name)

            coeffs = [None, 1, 0.5, 0.05, 0.005, 0.0005, 0.00005]
            plot_h = [h_list[int(len(h_list) / 4)], h_list[int(len(h_list) / 2)]]
            plt.loglog(plot_h, [coeffs[2] * h**2 for h in plot_h],
                       label="$\\mathcal{O}(\\Delta t^2)$", color="gray", marker="^")
            plt.loglog(plot_h, [coeffs[3] * h**3 for h in plot_h],
                       label="$\\mathcal{O}(\\Delta t^3)$", color="gray", marker="s")
            plt.loglog(plot_h, [coeffs[4] * h**4 for h in plot_h],
                       label="$\\mathcal{O}(\\Delta t^4)$", color="gray", marker="o")
            plot_h = [h_list[int(3 * len(h_list) / 8)], h_list[int(5 * len(h_list) / 8)]]
            plt.loglog(plot_h, [coeffs[5] * h**5 for h in plot_h],
                       label="$\\mathcal{O}(\\Delta t^5)$", color="gray", marker=">")
            plot_h = [h_list[int(len(h_list) / 2)], h_list[int(3 * len(h_list) / 4)]]
            plt.loglog(plot_h, [coeffs[6] * h**6 for h in plot_h],
                       label="$\\mathcal{O}(\\Delta t^6)$", color="gray", marker="<")
            plt.xlabel("$\\Delta t$")
            plt.ylabel("$|r(\\gamma = 1)|$")
            plt.legend()
            plt.grid()
            plt.savefig("P2_r_at_1.pdf")
            plt.show()

        end_point = 5

        def do_plot_3a():
            dotheplot(solvers_classic, test_problem2, exact_sol, end_point,
                      titles=["Regular SSPRK(3,3)", "Regular RK(4,4)", "Regular VRK(9,6)", "Regular VRK(13, 8)"],
                      filename="P2_Regular_RK4.pdf")

        def do_plot_3b():
            dotheplot(solvers_RRK, test_problem2, exact_sol, end_point,
                      titles=["Relaxed SSPRK(3,3)", "Relaxed RK(4,4)", "Relaxed VRK(9,6)", "Relaxed VRK(13, 8)"],
                      filename="P2_Relaxed_RK4.pdf")

        def do_plot_3c():
            dotheplot(solvers_IDT, test_problem2, exact_sol, end_point,
                      titles=["IDT SSPRK(3,3)", "IDT RK(4,4)", "IDT VRK(9,6)", "IDT VRK(13, 8)"],
                      filename="P2_IDT_RK4.pdf")

        def do_gamma_plot():
            RKtypes = ["SSPRK(3,3)", "RK(4,4)", "VRK(9,6)", "VRK(13, 8)"]
            RKfilenames = ["SSPRK33", "RK44", "VRK96", "VRK138"]
            stepsize = 0.01
            end_point = 20
            for i in range(len(solvers_classic)):
                approx = solvers_classic[i](test_problem2)
                approx._set_h(stepsize)
                t_list, y_list = approx.simulate(end_point)
                plt.semilogy(t_list[:-1], [np.abs(entropy2(y_list[j]) - entropy2(exact_sol(t_list[j]))) for j in range(len(t_list[:-1]))], label="Classic method")

                approx = solvers_RRK[i](test_problem2)
                approx._set_h(stepsize)
                t_list, y_list = approx.simulate(end_point)
                plt.semilogy(t_list[:-1], [np.abs(entropy2(y_list[j]) - entropy2(exact_sol(t_list[j]))) for j in range(len(t_list[:-1]))], label="RRK")

                approx = solvers_IDT[i](test_problem2)
                approx._set_h(stepsize)
                t_list, y_list = approx.simulate(end_point)
                plt.semilogy(t_list[:-1], [np.abs(entropy2(y_list[j]) - entropy2(exact_sol(t_list[j]))) for j in range(len(t_list[:-1]))], label="IDT")

                plt.title(f"Error in entropy of approximations\n of problem 2 using {RKtypes[i]} with $\\Delta t= {stepsize}$")
                plt.xlabel("Time, $t$")
                plt.ylabel("$|\\eta(u_i)-\\eta (u(t_i))|$")
                plt.legend()
                plt.grid()
                plt.savefig(f"P2_eta_error_{RKfilenames[i]}.pdf")
                plt.show()

        # do_plot_1a_2()
        # do_plot_1b_2()
        # do_plot_3a()
        # do_plot_3b()
        # do_plot_3c()
        do_gamma_plot()

    def do_problem_3():
        def func3(t, y):
            return np.array([-np.sin(y[1]), y[0]])

        def entropy3(y):
            return 0.5 * y[0]**2 - np.cos(y[1])

        def entropyprime3(y):
            return np.array([y[0], np.sin(y[1])])

        test_problem3 = EntropyProblem(np.array([1.5, 1]), func3, entropy3, entropyprime3)

        solver = IDT_SSPRK33(test_problem3)
        solver.simulate(5)
        solver.plot()

    do_problem_1()
    do_problem_2()
    # do_problem_3()
