sets_kfoldsets = [
    ['07AA002', '07QB002', '10JE001', '10DA001', '10KC001', '07FA004', '07FB002', '10GC001', '07AD002', '10JB001', '07ED001', '10LC014'],
    ['07CD001', '07OC001', '07SB019', '10HA004', '10AA001', '10KC001', '07FA004', '10MC002', '07HF001', '10EA003', '10LA002', '10BC001'],
    ['07CD001', '10EA003', '07OB003', '10LC014', '07EF001', '07NB001', '10GC001', '10BE001', '07QC003', '07GA001', '07KC001', '07FD003'],
    ['07CD005', '10BB001', '10EB001', '07CD001', '10BE001', '07FA004', '10LA002', '10AA001', '07LE003', '10BA001', '10KC001', '07JD002'],
    ['07EA005', '10LA002', '10BB002', '07TA001', '07EE007', '07FD002', '07FB003', '10JB001', '07EC002', '10MA001', '10ED002', '07SB019'],
    ['07EC002', '10GC001', '10BE001', '07FB003', '10MC002', '07OC001', '07TA001', '10ED002', '07KC001', '07EF001', '10BE006', '07EE007'],
    ['07EC002', '10HB005', '07HF001', '07FB006', '07OC001', '07JD002', '07CD005', '10GC001', '07FD002', '07EA005', '10EB001', '10AA001'],
    ['07ED001', '07AE001', '10MA001', '07QC003', '07FB003', '10AC002', '07EE007', '10LC014', '10AB001', '10DA001', '10EC001', '07FD010'],
    ['07EE007', '10BC001', '10LD004', '07EA005', '07FA004', '07EC002', '10HA004', '07FD003', '10BE006', '10ED002', '07FD002', '07QB002'],
    ['07EF001', '07AE001', '10DA001', '10AC002', '07HF001', '10AB001', '10KC001', '10MC002', '10JE001', '07FC001', '10HB005', '10EB001'],
    ['07FB003', '07FB006', '07FC001', '10FB006', '10BB002', '07FD003', '10HA003', '10BC001', '10AB001', '10LC014', '07EA005', '07CD001'],
    ['07FB006', '10BB002', '07GA001', '10GC001', '07CD001', '10BE010', '07QC007', '07EE007', '07EC002', '07JD002', '10MA001', '10JB001'],
    ['07FC001', '07SA005', '10DA001', '10EB001', '07FB006', '07TA001', '07AE001', '07LE003', '07QC007', '07SB019', '07FB003', '07KC001'],
    ['07FD002', '10KC001', '07KC001', '10MA001', '07CD001', '10MC002', '07FD003', '10BE010', '10LC014', '07GA001', '10HA004', '07JD002'],
    ['07FD010', '07AD002', '10HB005', '07QC007', '10BE001', '07QC005', '07OB003', '07AE001', '07AA002', '07SB019', '10AD001', '10DA001'],
    ['07GA001', '07FA004', '10HA004', '10LA002', '10LD004', '07FC001', '07FD010', '07QC003', '07FD003', '07CD001', '10BC001', '07QB002'],
    ['07KC001', '10BC001', '10AC002', '10LA002', '07GA001', '10BE001', '10AD001', '10BE010', '07FB002', '07EE007', '10HB005', '07LE003'],
    ['07KC001', '10BE010', '10BC001', '07FB002', '07FD003', '07SB019', '07HF001', '10ED002', '07AD002', '07QC005', '10BA001', '10KC001'],
    ['07LE003', '10BE006', '07EA005', '07AD002', '10GC001', '07FC001', '07FD010', '07FB006', '10HB005', '07GA001', '07JD002', '07HF001'],
    ['07LE003', '10DA001', '07KC001', '10BC001', '10ED001', '10EB001', '10LA002', '07FB002', '07EE007', '07JD002', '10HA003', '07AE001'],
    ['07NB001', '10BB001', '10BC001', '07SB019', '07ED001', '07HF001', '10FB006', '10EC001', '07QC003', '07FC001', '07SA005', '07OC001'],
    ['07QB002', '07CD001', '07FB006', '07SB019', '10GC001', '10ED002', '07FD002', '07FD010', '07LE003', '10AC002', '07KC001', '07FC001'],
    ['07QB002', '07QC003', '10CD001', '07AD002', '07LE003', '07EE007', '10GC001', '10EB001', '10JE001', '07KC001', '07CD001', '10HA003'],
    ['07QC003', '07ED001', '07EE007', '10DA001', '10BA001', '10BE006', '10JB001', '07CD001', '07FB003', '07FA004', '10BB002', '10JE001'],
    ['07QC005', '07QC003', '10BA001', '10EC001', '10AC002', '10MC002', '10EB001', '07SB019', '07AE001', '10LA002', '07EE007', '10FB006'],
    ['07QD007', '10AC002', '07EF001', '07FB003', '07FC001', '10MC002', '10AB001', '07FB002', '07EC002', '10BB002', '07LE003', '07QC005'],
    ['07QD007', '10FB006', '10EB001', '10GC001', '10ED001', '07EE007', '10ED002', '07EA005', '10HA003', '10EA003', '10BA001', '10AD001'],
    ['07SA005', '10JB001', '10AC002', '07QB002', '10KC001', '07EC002', '07QC005', '10AB001', '10EA003', '10EC001', '07LE003', '10FB006'],
    ['07SB019', '10FB006', '07FA004', '07TA001', '07FC001', '07FB006', '10DA001', '10AD001', '10AB001', '10BE006', '10LA002', '10MC002'],
    ['10AB001', '07LE003', '07SB019', '10ED002', '07QC005', '07QD007', '07CD001', '07TA001', '10AD001', '10BE006', '07QB002', '10BE010'],
    ['10AB001', '10KC001', '07QC005', '07QD007', '10FB006', '10HA003', '07OB003', '10ED002', '10LD004', '10BB002', '07FD010', '07QC003'],
    ['10AD001', '10AB001', '07EA005', '10BA001', '10MA001', '07QC007', '07CD005', '10ED002', '07SB019', '07FD002', '10LA002', '10BB001'],
    ['10BA001', '07TA001', '10HA004', '07FB003', '10JE001', '07JD002', '10MA001', '10GC001', '07FD003', '10BE005', '10ED001', '07AE001'],
    ['10BB002', '07CD001', '10AB001', '10GC001', '10MA001', '10DA001', '07OC001', '10BB001', '10HA003', '07HF001', '10FB006', '07EA005'],
    ['10BC001', '07QC003', '07HF001', '10HA003', '10JB001', '10ED002', '10BE001', '07QD007', '10AD001', '10GC001', '10BE006', '07KC001'],
    ['10BC001', '10DA001', '07EE007', '10LA002', '10BE001', '10EB001', '07FD002', '07QB002', '07OB003', '10BB002', '07KC001', '10GC001'],
    ['10BE005', '07GA001', '10AA001', '07NB001', '07HF001', '10BB002', '10CD001', '07QD007', '07SB019', '07EF001', '07AE001', '07FA004'],
    ['10BE005', '07OB003', '07JD002', '10KC001', '10BA001', '10LD004', '07EE007', '07SA005', '07FD010', '07TA001', '07QC005', '07AE001'],
    ['10CD001', '10BB002', '10ED001', '07QC007', '07HF001', '10FB006', '07QB002', '07QD007', '07OB003', '07NB001', '07AD002', '07FB002'],
    ['10DA001', '07CD005', '10AC002', '10EA003', '10EB001', '07KC001', '07FD003', '10FB006', '07AD002', '10HA003', '07EF001', '07OB003'],
    ['10EA003', '07OC001', '07FA004', '10JE001', '10KC001', '10CD001', '07FD003', '07QC005', '10BE010', '07NB001', '10HA003', '10EB001'],
    ['10EA003', '10AD001', '10FB006', '10HA004', '07AE001', '10AC002', '07SA005', '10JE001', '10DA001', '07OC001', '07GA001', '07FD010'],
    ['10EC001', '07EE007', '10AD001', '10LA002', '10HA004', '07GA001', '07FD002', '10BB002', '10BE010', '10LD004', '10BA001', '10LC014'],
    ['10HA003', '07OB003', '10DA001', '07JD002', '10AC002', '10ED002', '07SB019', '07OC001', '10ED001', '07AE001', '10EC001', '10LC014'],
    ['10HA004', '07SB019', '07OC001', '10LD004', '07SA005', '10MC002', '07FA004', '10BB002', '10JB001', '10JE001', '07QC005', '07FB003'],
    ['10JB001', '10EA003', '10ED002', '10HA003', '10AD001', '10MA001', '07QC003', '07EF001', '07OC001', '07NB001', '10AC002', '10DA001'],
    ['10LA002', '07LE003', '07GA001', '10FB006', '07QC003', '10BB001', '10BE001', '07OC001', '10DA001', '10BC001', '07SA005', '07FB002'],
    ['10LC014', '07JD002', '07QC007', '07FC001', '10BE005', '07QC005', '07FD010', '07EC002', '10HA004', '07AE001', '07FA004', '10BC001'],
    ['10LD004', '10HA004', '07EF001', '10BC001', '07KC001', '10MA001', '10GC001', '10CD001', '07CD005', '10AD001', '07NB001', '10LA002'],
    ['10MC002', '10MA001', '10HA004', '10GC001', '07CD001', '07JD002', '07EA005', '10BB001', '07TA001', '07KC001', '10HA003', '07FC001']
]

sets_1m2rta = [
    [82042777, 82017021, 82005809, 82023555, 82008607, 82037818, 82038875, 82018072, 82042445, 82010991, 82039917, 82003287],
    [82035594, 82029883, 82013426, 82011035, 82023560, 82008607, 82037818, 82004055, 82029872, 82019559, 82004876, 82025133],
    [82035594, 82019559, 82028215, 82003287, 82037835, 82025015, 82018072, 82026688, 82022245, 82042094, 82028201, 82037779],
    [82036808, 82025121, 82019471, 82035594, 82026688, 82037818, 82004876, 82023560, 82026787, 82028331, 82008607, 82031602],
    [82034349, 82004876, 82028370, 82014602, 82039898, 82037816, 82038962, 82010991, 82037880, 82006612, 82018092, 82013426],
    [82037880, 82018072, 82026688, 82038962, 82004055, 82029883, 82014602, 82018092, 82028201, 82037835, 82025069, 82039898],
    [82037880, 82012264, 82029872, 82039882, 82029883, 82031602, 82036808, 82018072, 82037816, 82034349, 82019471, 82023560],
    [82039917, 82041573, 82006612, 82022245, 82038962, 82026945, 82039898, 82003287, 82022225, 82023555, 82019447, 82037800],
    [82039898, 82025133, 82005753, 82034349, 82037818, 82037880, 82011035, 82037779, 82025069, 82018092, 82037816, 82017021],
    [82037835, 82041573, 82023555, 82026945, 82029872, 82022225, 82008607, 82004055, 82005809, 82036797, 82012264, 82019471],
    [82038962, 82039882, 82036797, 82018084, 82028370, 82037779, 82011128, 82025133, 82022225, 82003287, 82034349, 82035594],
    [82039882, 82028370, 82042094, 82018072, 82035594, 82026827, 82023517, 82039898, 82037880, 82031602, 82006612, 82010991],
    [82036797, 82014701, 82023555, 82019471, 82039882, 82014602, 82041573, 82026787, 82023517, 82013426, 82038962, 82028201],
    [82037816, 82008607, 82028201, 82006612, 82035594, 82004055, 82037779, 82026827, 82003287, 82042094, 82011035, 82031602],
    [82037800, 82042445, 82012264, 82023517, 82026688, 82025061, 82028215, 82041573, 82042777, 82013426, 82025158, 82023555],
    [82042094, 82037818, 82011035, 82004876, 82005753, 82036797, 82037800, 82022245, 82037779, 82035594, 82025133, 82017021],
    [82028201, 82025133, 82026945, 82004876, 82042094, 82026688, 82025158, 82026827, 82038875, 82039898, 82012264, 82026787],
    [82028201, 82026827, 82025133, 82038875, 82037779, 82013426, 82029872, 82018092, 82042445, 82025061, 82028331, 82008607],
    [82026787, 82025069, 82034349, 82042445, 82018072, 82036797, 82037800, 82039882, 82012264, 82042094, 82031602, 82029872],
    [82026787, 82023555, 82028201, 82025133, 82023524, 82019471, 82004876, 82038875, 82039898, 82031602, 82011128, 82041573],
    [82025015, 82025121, 82025133, 82013426, 82039917, 82029872, 82018084, 82019447, 82022245, 82036797, 82014701, 82029883],
    [82017021, 82035594, 82039882, 82013426, 82018072, 82018092, 82037816, 82037800, 82026787, 82026945, 82028201, 82036797],
    [82017021, 82022245, 82028307, 82042445, 82026787, 82039898, 82018072, 82019471, 82005809, 82028201, 82035594, 82011128],
    [82022245, 82039917, 82039898, 82023555, 82028331, 82025069, 82010991, 82035594, 82038962, 82037818, 82028370, 82005809],
    [82025061, 82022245, 82028331, 82019447, 82026945, 82004055, 82019471, 82013426, 82041573, 82004876, 82039898, 82018084],
    [82022185, 82026945, 82037835, 82038962, 82036797, 82004055, 82022225, 82038875, 82037880, 82028370, 82026787, 82025061],
    [82022185, 82018084, 82019471, 82018072, 82023524, 82039898, 82018092, 82034349, 82011128, 82019559, 82028331, 82025158],
    [82014701, 82010991, 82026945, 82017021, 82008607, 82037880, 82025061, 82022225, 82019559, 82019447, 82026787, 82018084],
    [82013426, 82018084, 82037818, 82014602, 82036797, 82039882, 82023555, 82025158, 82022225, 82025069, 82004876, 82004055],
    [82022225, 82026787, 82013426, 82018092, 82025061, 82022185, 82035594, 82014602, 82025158, 82025069, 82017021, 82026827],
    [82022225, 82008607, 82025061, 82022185, 82018084, 82011128, 82028215, 82018092, 82005753, 82028370, 82037800, 82022245],
    [82025158, 82022225, 82034349, 82028331, 82006612, 82023517, 82036808, 82018092, 82013426, 82037816, 82004876, 82025121],
    [82028331, 82014602, 82011035, 82038962, 82005809, 82031602, 82006612, 82018072, 82037779, 82025057, 82023524, 82041573],
    [82028370, 82035594, 82022225, 82018072, 82006612, 82023555, 82029883, 82025121, 82011128, 82029872, 82018084, 82034349],
    [82025133, 82022245, 82029872, 82011128, 82010991, 82018092, 82026688, 82022185, 82025158, 82018072, 82025069, 82028201],
    [82025133, 82023555, 82039898, 82004876, 82026688, 82019471, 82037816, 82017021, 82028215, 82028370, 82028201, 82018072],
    [82025057, 82042094, 82023560, 82025015, 82029872, 82028370, 82028307, 82022185, 82013426, 82037835, 82041573, 82037818],
    [82025057, 82028215, 82031602, 82008607, 82028331, 82005753, 82039898, 82014701, 82037800, 82014602, 82025061, 82041573],
    [82028307, 82028370, 82023524, 82023517, 82029872, 82018084, 82017021, 82022185, 82028215, 82025015, 82042445, 82038875],
    [82023555, 82036808, 82026945, 82019559, 82019471, 82028201, 82037779, 82018084, 82042445, 82011128, 82037835, 82028215],
    [82019559, 82029883, 82037818, 82005809, 82008607, 82028307, 82037779, 82025061, 82026827, 82025015, 82011128, 82019471],
    [82019559, 82025158, 82018084, 82011035, 82041573, 82026945, 82014701, 82005809, 82023555, 82029883, 82042094, 82037800],
    [82019447, 82039898, 82025158, 82004876, 82011035, 82042094, 82037816, 82028370, 82026827, 82005753, 82028331, 82003287],
    [82011128, 82028215, 82023555, 82031602, 82026945, 82018092, 82013426, 82029883, 82023524, 82041573, 82019447, 82003287],
    [82011035, 82013426, 82029883, 82005753, 82014701, 82004055, 82037818, 82028370, 82010991, 82005809, 82025061, 82038962],
    [82010991, 82019559, 82018092, 82011128, 82025158, 82006612, 82022245, 82037835, 82029883, 82025015, 82026945, 82023555],
    [82004876, 82026787, 82042094, 82018084, 82022245, 82025121, 82026688, 82029883, 82023555, 82025133, 82014701, 82038875],
    [82003287, 82031602, 82023517, 82036797, 82025057, 82025061, 82037800, 82037880, 82011035, 82041573, 82037818, 82025133],
    [82005753, 82011035, 82037835, 82025133, 82028201, 82006612, 82018072, 82028307, 82036808, 82025158, 82025015, 82004876],
    [82004055, 82006612, 82011035, 82018072, 82035594, 82031602, 82034349, 82025121, 82014602, 82028201, 82011128, 82036797]
]