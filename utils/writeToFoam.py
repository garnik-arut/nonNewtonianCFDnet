# import numpy as np


def time_step(ux, uy, u_avg, p, pplus, nut, nut_avg, ux_top, uy_top, p_top, nut_top, case, turbulence):
    if case == "ellipse":
        # VELOCITY
        s = ux.shape[0]
        s = str(s)
        f = open("./printedFields/U", "w")
        f.write(
            "FoamFile\n{\n	version 	2.0;\n	format	ascii;\n  class volVectorField;\n	location	0;\n	object	U;\n}\n")
        f.write("dimensions [0 1 -1 0 0 0 0];\n\n")
        f.write("internalField\t nonuniform List<vector>\n" + s + "\n(\n")
        for j in range(0, int(s)):
            f.write("(" + repr(ux[j] * u_avg) + " 0 " + repr(uy[j] * u_avg) + ")\n")
        f.write(");\n")
        f.write("boundaryField\n{\n")
        f.write("\ntop\n{\n\ttype\t freestream;\n\tfreestreamValue\t uniform (0.6 0 0);\n}")
        f.write("\nbottom\n{\n\ttype\t noSlip;\n}")
        f.write("\nfrontAndBack\n{\n\ttype\t empty;\n}\n")
        f.write("}\n")

        # PRESSURE
        s = ux.shape[0]
        s = str(s)
        f = open("./printedFields/p", "w")
        f.write(
            "FoamFile\n{\n	version 	2.0;\n	format	ascii;\n  class volScalarField;\n	location	0;\n	object	p;\n}\n")
        f.write("dimensions [0 2 -2 0 0 0 0];\n\n")
        f.write("internalField\t nonuniform List<scalar>\n" + s + "\n(\n")
        for j in range(0, int(s)):
            f.write(repr(p[j] * (u_avg * u_avg) / pplus) + "\n")

        f.write(");\n")
        f.write("boundaryField\n{\n")
        f.write("\ntop\n{\n\ttype\t freestream;\n\tfreestreamValue\t uniform 0;\n}")
        f.write("\nbottom\n{\n\ttype\t zeroGradient;\n}")
        f.write("\nfrontAndBack\n{\n\ttype\t empty;\n}\n")

        f.write("}\n")

        if turbulence == 1:
            s = ux.shape[0]
            s = str(s)
            f = open("./printedFields/nuTilda", "w")
            f.write(
                "FoamFile\n{\n	version 	2.0;\n	format	ascii;\n  class volScalarField;\n	location	0;\n	"
                "object	nuTilda;\n}\n")
            f.write("dimensions [0 2 -1 0 0 0 0];\n\n")
            f.write("internalField\t nonuniform List<scalar>\n" + s + "\n(\n")
            for j in range(0, int(s)):
                f.write(repr(nut[j] * nut_avg) + "\n")

            f.write(");\n")
            s = str(s)
            f.write("boundaryField\n{\n")
            f.write("\ntop\n{\n\ttype\t freestream;\n\tfreestreamValue\t uniform 3e-6;\n}")

            f.write("\nbottom\n{\n\ttype\t fixedValue;\nvalue\t uniform 0;}")
            f.write("\nfrontAndBack\n{\n\ttype\t empty;\n}\n")

            f.write("}\n")

    if case == "channelFlow":

        # VELOCITY

        s = ux.shape[0]
        s = str(s)
        f = open("./printedFields/U", "w")
        f.write(
            "FoamFile\n{\n	version 	2.0;\n	format	ascii;\n  class volVectorField;\n	location	0;\n	object	U;\n}\n")
        f.write("dimensions [0 1 -1 0 0 0 0];\n\n")
        f.write("internalField\t nonuniform List<vector>\n" + s + "\n(\n")
        for j in range(0, int(s)):
            f.write("(" + repr(ux[j] * u_avg) + " " + repr(uy[j] * u_avg) + " 0)\n")

        f.write(");\n")

        f.write("boundaryField\n{\n")
        f.write("inlet\n{\n\ttype\t fixedValue;\n\tvalue\t uniform(" + repr(u_avg) + " 0 0);\n}")
        f.write("\noutlet\n{\n\ttype\t zeroGradient;\n}")
        f.write("\ntop\n{\n\ttype\t noSlip;\n}")
        f.write("\nbottom\n{\n\ttype\t noSlip;\n}")
        f.write("\nfrontAndBack\n{\n\ttype\t empty;\n}\n")

        f.write("}\n")

        # PRESSURE

        s = ux.shape[0]
        s = str(s)
        f = open("./printedFields/p", "w")
        f.write(
            "FoamFile\n{\n	version 	2.0;\n	format	ascii;\n  class volScalarField;\n	location	0;\n	object	p;\n}\n")
        f.write("dimensions [0 2 -2 0 0 0 0];\n\n")
        f.write("internalField\t nonuniform List<scalar>\n" + s + "\n(\n")
        for j in range(0, int(s)):
            f.write(repr(p[j] * (u_avg * u_avg) / pplus) + "\n")

        f.write(");\n")

        f.write("boundaryField\n{\n")
        f.write("outlet\n{\n\ttype\t fixedValue;\n\tvalue\t uniform 0;\n}")
        f.write("\ninlet\n{\n\ttype\t zeroGradient;\n}")
        f.write("\ntop\n{\n\ttype\t zeroGradient;\n}")
        f.write("\nbottom\n{\n\ttype\t zeroGradient;\n}")
        f.write("\nfrontAndBack\n{\n\ttype\t empty;\n}\n")

        f.write("}\n")

        if turbulence == 1:

            s = ux.shape[0]
            s = str(s)
            f = open("./printedFields/nuTilda", "w")
            f.write(
                "FoamFile\n{\n	version 	2.0;\n	format	ascii;\n  class volScalarField;\n	location	0;\n	"
                "object	nuTilda;\n}\n")
            f.write("dimensions [0 2 -1 0 0 0 0];\n\n")
            f.write("internalField\t nonuniform List<scalar>\n" + s + "\n(\n")
            for j in range(0, int(s)):
                f.write(repr(nut[j] * nut_avg) + "\n")

            f.write(");\n")

            f.write("boundaryField\n{\n")
            f.write("inlet\n{\n\ttype\t fixedValue;\n\tvalue\t uniform 0.001;\n}")
            f.write("\noutlet\n{\n\ttype\t zeroGradient;\n}")
            f.write("\ntop\n{\n\ttype\t fixedValue;\n\tvalue\t uniform 0;\n}")
            f.write("\nbottom\n{\n\ttype\t fixedValue;\n\tvalue\t uniform 0;\n}")
            f.write("\nfrontAndBack\n{\n\ttype\t empty;\n}\n")

            f.write("}\n")
