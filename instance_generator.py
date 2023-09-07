import os
import numpy as np
from random import sample, randint, uniform, choice, choices,seed
from rosters import generate_roster
from math import ceil, floor
import codecs
import json
import sys

def generate_shift_data(shifts):
    shift_data = {}

    for shift in shifts:
        if shift % 3 == 1:
            walk_path_circle = round(uniform(0.7, 0.9), 2)
            walk_path_star = round(1 - walk_path_circle, 2)
        elif shift % 3 == 2:
            walk_path_circle = round(uniform(0.4, 0.6), 2)
            walk_path_star = round(1 - walk_path_circle, 2)
        else:
            assert shift % 3 == 0
            walk_path_circle = round(uniform(0.1, 0.3), 2)
            walk_path_star = round(1 - walk_path_circle, 2)

        shift_data[shift] = {
            "circleWeight": walk_path_circle,
            "starWeight": walk_path_star,
        }
    return shift_data


def generate_nurses_max_load(skill_level, shifts, skill_max_load):
    try:
        return {shift: skill_max_load[skill_level] for shift in shifts}
    except IndexError:
        print(f"An error occurred: Index out of range. The skill_level value {skill_level} does not fit the amount of skill_max_load entries ({len(skill_max_load)} entries). Please check your input data.")
        sys.exit(1)

def generate_nurse_roster(
    nurses, shifts, nurse_distribution_per_shift, max_shifts_per_nurse
):
    nDays = int(len(shifts) / len(nurse_distribution_per_shift))
    return generate_roster(
        nurses,
        shifts,
        nDays * nurse_distribution_per_shift,
        max_shifts_per_nurse,
    )


def generate_nurse_vary_skills(nurses, n, n_nurses_of_level):
    for level in range(len(n_nurses_of_level)):
        for i in range(n_nurses_of_level[level]):
            nurses.append(
                {
                    "id": str(n),
                    "skillLevel": level,
                }
            )
            n += 1

    # Ensure the correct number of nurses have been generated
    assert len(nurses) == sum(
        n_nurses_of_level
    ), "something went wrong in the generation of nurses' skill levels"
    return nurses, n


def generate_nurse_data(
    shifts,
    nurse_distribution_per_shift,
    max_shifts_per_nurse,
    fix_nurses,
    nurse_skill_level,
    nurse_max_load,
    mode,
):
    n = 0
    nurses = []
    if mode == "auto":
        n_nurses_of_level = nurse_distribution_to_skill_set(
            fix_nurses, nurse_skill_level
        )
        nurses, n = generate_nurse_vary_skills(nurses, n, n_nurses_of_level)
        # Generate nurse rosters and assign maximum load
        try:
            nurses = generate_nurse_roster(
                nurses,
                shifts,
                nurse_distribution_per_shift,
                max_shifts_per_nurse
            )
        except ValueError as v_er:
            print("Error: %s" % v_er)
            nurses, fix_nurses = generate_nurse_data(
                shifts,
                nurse_distribution_per_shift,
                max_shifts_per_nurse,
                fix_nurses + 1,
                nurse_skill_level,
                nurse_max_load,
                mode,
            )
        for n in nurses:
            n["maxLoad"] = generate_nurses_max_load(n["skillLevel"],n["workingShifts"], nurse_max_load)
        return nurses, fix_nurses
    elif mode == "manual":
        n_nurses_of_level = nurse_distribution_to_skill_set(
            fix_nurses, nurse_skill_level
        )
        nurses, n = generate_nurse_vary_skills(nurses, n, n_nurses_of_level)
        # Generate nurse rosters and assign maximum load
        nurses = generate_nurse_roster(
            nurses,
            shifts,
            nurse_distribution_per_shift,
            max_shifts_per_nurse
        )
        for n in nurses:
            generate_nurses_max_load(n["skilllevel"],n["workingShifts"], nurse_max_load)

        return nurses, fix_nurses


def gen_room_data(room_config, equipment):
    # Generate a range of room names based on room capacities
    roomName = room_config[0]  # Initialize room name counter
    rooms = []  # List to store room data

    # Iterate through different room capacities
    for cap in range(len(room_config[1:])):
        for i in range(room_config[cap + 1]):
            # Determine the equipment for the room
            if equipment:
                eq = list(sample(list(equipment), randint(0, len(equipment))))
            else:
                eq = []  # No equipment if E is not provided

            # Create a dictionary representing room information
            rooms.append(
                {
                    "id": str(roomName),  # Assign unique room ID
                    "capacity": cap + 1,  # Assign room capacity
                    "equipment": eq,  # Assign equipment list
                }
            )
            roomName += 1  # Increment room name counter
    return rooms  # Return the list of generated room data


def calculate_rooms_distribution(room_size, room_balancing, rest_room):
    if len(room_balancing) != 4:
        raise ValueError("Percentages array must contain exactly 4 values.")

    total_percentage = sum(room_balancing)
    if total_percentage != 100:
        raise ValueError("Total percentage must be 100.")
    room_counts = [int(round(room_size * p / 100,0)) for p in room_balancing]
    remaining_rooms = room_size - sum(room_counts)

    # Distribute remaining rooms to the first non-zero percentage
    for i in range(len(room_counts)):
        if remaining_rooms == 0:
            break
        if room_balancing[i] != 0:
            room_counts[i] += 1
            remaining_rooms -= 1
    room_counts.insert(0, rest_room)
    return room_counts


def total_needed_nurses_per_shift(room_config):
    t_capacity = sum(i * room_config[i] for i in range(len(room_config)))
    # Nurses per shift:
    #      [8 patients per nurse, 8 patients per nurse, 12 patients per nurse]
    return [ceil(t_capacity / 8), ceil(t_capacity / 8), ceil(t_capacity / 12)]


# Compute nurse distribution to skill set
def nurse_distribution_to_skill_set(nNurses, nSkills):
    assert 1 <= nSkills <= 3
    if nSkills == 3:
        return [
            nNurses - ceil(0.2 * nNurses) - floor(0.6 * nNurses),
            floor(0.6 * nNurses),
            ceil(0.2 * nNurses),
        ]
    if nSkills == 2:
        return [nNurses - ceil(0.8 * nNurses), ceil(0.8 * nNurses)]
    assert nSkills == 1
    return [nNurses]


def generate_patient_los(los_param):
    assert len(los_param) >= 2, "missing Parameters for los distribution"
    assert los_param[0] <= los_param[1], "los parameters invalid"
    return randint(los_param[0], los_param[1])


def generate_patient_skill_requirements(ad_shift, di_shift,
                                        skills, n_shifts_per_day):
    # Monotonously decreasing skill level requirements for each shift type
    skill_requirements = {}
    for s in range(ad_shift, di_shift + 1):
        if s - ad_shift > n_shifts_per_day:
            skill_requirements[s] = min(
                skill_requirements[s - n_shifts_per_day], choice(skills)
            )
        else:
            skill_requirements[s] = choice(skills)
    return skill_requirements


def generate_patient_gender(gender_list):
    return choices(
        gender_list["genderList"],
        weights=gender_list["probabilityList"], k=1)[0]


def generate_start_assignment(nurses, rooms):
    return [], None, 0


def generate_patient_equipment_requirement(ad_shift, di_shift, equipment):
    # Monotonously decreasing requirements
    if not equipment:
        eq_req = {ad_shift: []}
    else:
        eq_req = {ad_shift: list(sample(
                            list(equipment),
                            randint(1, len(equipment))
                            ))}
    for s in range(ad_shift + 1, di_shift + 1):
        eq_req[s] = list(sample(eq_req[s - 1], randint(0, len(eq_req[s - 1]))))
    return eq_req


def generate_patient_workload(work_load_distribution, age_group,
                              ad_shift, di_shift):
    min_bound = work_load_distribution[2]
    max_bound = work_load_distribution[3]
    q = work_load_distribution[4]
    initial_raw_workload = round(
        np.random.gamma(
            work_load_distribution[0],
            work_load_distribution[1] + age_group / 10
        )
    )
    initial_workload = (
        min_bound
        if initial_raw_workload < min_bound
        else max_bound
        if initial_raw_workload > max_bound
        else initial_raw_workload
    )

    # Monotonously decreasing work load for each shift type
    wLoad = {}
    for shift in range(ad_shift, di_shift + 1):
        if shift - ad_shift < 3 and not shift % 3 == 0:
            wLoad[shift] = initial_workload
        elif shift - ad_shift < 3 and shift % 3 == 0:
            wLoad_shift_night = int(initial_workload / 3)
            wLoad[shift] = wLoad_shift_night if wLoad_shift_night > 0 else 1
        else:
            wLoad_shift = int(wLoad[shift - 3]
                * (1 - q)
                ** (shift - ad_shift))
            wLoad[shift] = wLoad_shift if wLoad_shift > 0 else 1
    return wLoad


def generate_patient_data(
    rooms,
    gender_list,
    shifts,
    nurses,
    equipment,
    skills,
    n_shifts_per_day,
    work_load_distribution,
    los_params,
    occupancy_rate,
):
    # Count rooms with specific equipment
    def count_rooms_with_equipment(rooms, e):
        return len([r["id"] for r in rooms if e in r["equipment"]])

    # Get equipment availability per shift
    def get_equipment_per_shift(rooms, equipment, shifts):
        return {
            s: {e: count_rooms_with_equipment(rooms, e) for e in equipment}
            for s in shifts
        }

    # Get available equipment set
    def get_available_equipment_set(available_equipment):
        return {
            e for e in available_equipment.keys()
            if available_equipment[e] > 0
            }

    patient_data = []
    available_equip_all_shifts = get_equipment_per_shift(
                                                        rooms, equipment,
                                                        shifts)

    # Generate patient data for each room based on occupancy rate
    for r in range(int(occupancy_rate * sum(r["capacity"] for r in rooms))):
        ad_shift = 1
        while ad_shift < max(shifts):
            p = {}
            p["id"] = str(len(patient_data) + 1)
            p["ageGroup"] = randint(3, 10)
            p["admission"] = ad_shift
            p["discharge"] = min(
                p["admission"]
                + n_shifts_per_day * generate_patient_los(los_params)
                - 1,
                max(shifts),
            )
            p["gender"] = generate_patient_gender(gender_list)
            p["skillReq"] = generate_patient_skill_requirements(
                p["admission"], p["discharge"], skills, n_shifts_per_day
            )
            p["workLoad"] = generate_patient_workload(
                work_load_distribution,
                p["ageGroup"],
                p["admission"],
                p["discharge"],
            )
            p["equipmentReq"] = generate_patient_equipment_requirement(
                p["admission"],
                p["discharge"],
                get_available_equipment_set(
                    available_equip_all_shifts[ad_shift]
                    ),
            )
            for s in p["equipmentReq"].keys():
                for e in p["equipmentReq"][s]:
                    available_equip_all_shifts[s][e] -= 1
            (
                p["prevAssignedNurses"],
                p["currentRoom"],
                p["prevTransfers"],
            ) = generate_start_assignment(nurses, rooms)
            patient_data.append(p)
            ad_shift = p["discharge"] + 1
    return patient_data


def get_distance(id_start, id_end):
    if id_start == id_end:
        return 0
    return 2.8 + 5 + 4 * floor(abs(id_start - id_end) / 2.0)


def generate_instance(
    weekdays,
    max_nurse_shifts_per_week,
    work_load_distribution,
    los_params,
    fix_nurses,
    patient_gender_distribution,
    room_config,
    room_balancing,
    nurse_skill_level,
    nurse_max_load,
    equipment,
    n_week,
    occupancy_rate,
    inst_id,
    mode,
):
    instance_data = {}
    assert 0 <= occupancy_rate <= 1, "Occupancy rate must be between 0 and 1"
    assert (
        len(room_config) == 5
        and sum(room_config[1:]) > 0
        and room_config[0] > 0
    ), "Check room instructions"
    required_nurses_per_shift = total_needed_nurses_per_shift(room_config)
    print(required_nurses_per_shift)
    nurse_distribution_per_shift = [
        nurse_distribution_to_skill_set(a, nurse_skill_level)
        for a in required_nurses_per_shift
    ]
    print(nurse_distribution_per_shift)
    n_shifts_per_day = len(nurse_distribution_per_shift)
    shifts = range(1, n_shifts_per_day * weekdays * n_week + 1)
    assert n_shifts_per_day == 3, "Only 3 Shifts per day allowed"
    assert len(work_load_distribution) == 5, "workload computation needs 5"
    instance_data["days"] = {
        "firstDay": 1,
        "lastDay": int(weekdays) * int(n_week),
    }
    instance_data["equipment"] = list(equipment)
    instance_data["rooms"] = gen_room_data(room_config, equipment)
    additional_rooms = [{"id": str(r)} for r in range(room_config[0])]
    instance_data["additionalRooms"] = list(additional_rooms)
    instance_data["shifts"] = generate_shift_data(shifts)
    instance_data["skillLevels"] = list(range(nurse_skill_level))
    nurses, fix_nurses = generate_nurse_data(
        shifts,
        nurse_distribution_per_shift,
        int(max_nurse_shifts_per_week) * int(n_week),
        fix_nurses=fix_nurses,
        nurse_skill_level=nurse_skill_level,
        nurse_max_load=nurse_max_load,
        mode=mode,
    )
    instance_data["nurses"] = nurses
    instance_data["patients"] = generate_patient_data(
        instance_data["rooms"],
        patient_gender_distribution,
        shifts,
        instance_data["nurses"],
        equipment,
        instance_data["skillLevels"],
        n_shifts_per_day,
        work_load_distribution,
        los_params,
        occupancy_rate,
    )
    instance_data["distances"] = {
        r["id"]: {
            s["id"]: get_distance(int(r["id"]), int(s["id"]))
            for s in additional_rooms + instance_data["rooms"]
        }
        for r in additional_rooms + instance_data["rooms"]
    }
    os.makedirs(
        "instances",
        exist_ok=True,
    )
    # Write instance to file
    instance_name = "_".join(
        [
            str(n_week),
            str(sum(room_config[1:])),
            str("(" + str(" ".join(str(bal) for bal in room_balancing)) + ")"),
            str(fix_nurses),
            str(nurse_skill_level),
            str(len(equipment)),
            str(inst_id),
        ]
    )
    name = (
        "instances/random_instance_"
        + str(instance_name)
        + ".json"
    )
    with codecs.open(name, "w", encoding="utf-8") as out:
        out.write(json.dumps(instance_data, indent=4))
    return instance_name


def generate_instance_set(
    n_eq_instances,
    plan_weeks,
    room_sizes,
    room_balances,
    equipments,
    nurse_skills,
    nurse_max_load,
    rest_rooms,
    occupancy_rates=[0.95],
    fix_nurses=21,
    mode="auto",
):
    # additional Params
    weekdays = 7
    max_nurse_shifts_per_week = 5
    work_load_distribution = [3, 0.5, 1, 5, 0.1]
    los_params = [1, 5]
    patient_gender_distribution = {
        "genderList": ["F", "M"],
        "probabilityList": [0.5, 0.5]
        }
    for room_size in room_sizes:
        for room_balancing in room_balances:
            for rest_room in rest_rooms:
                room_config = calculate_rooms_distribution(
                    room_size, room_balancing, rest_room
                )

                for nurse_skill_level in nurse_skills:
                    for equipment in equipments:
                        for n_week in plan_weeks:
                            for occupancy_rate in occupancy_rates:
                                for inst_id in range(n_eq_instances):
                                    instance = generate_instance(
                                        weekdays,
                                        max_nurse_shifts_per_week,
                                        work_load_distribution,
                                        los_params,
                                        fix_nurses,
                                        patient_gender_distribution,
                                        room_config,
                                        room_balancing,
                                        nurse_skill_level,
                                        nurse_max_load,
                                        equipment,
                                        n_week,
                                        occupancy_rate,
                                        inst_id,
                                        mode,
                                    )
                                    print(instance, "successfully")

if __name__ == "__main__":
    seed(1208)
    # Example 1 (cf. Brandt et al. 20230 Variation 1)
    gen = generate_instance_set(
        n_eq_instances=10,
        plan_weeks=[2, 4],
        room_sizes=[15,30],
        room_balances=[[0, 100, 0, 0]],
        equipments=[{"oxygen", "telemetry"}],
        nurse_skills=[3],
        nurse_max_load = [10, 12.5, 15],
        rest_rooms=[1],
        occupancy_rates=[0.85],
        fix_nurses=21,
        mode="auto",
    )
    # Example 2 (cf. Brandt et al. 20230 Variation 2)
    gen = generate_instance_set(
        n_eq_instances=10,
        plan_weeks=[2, 4],
        room_sizes=[10,20],
        room_balances=[[0, 0, 100, 0]],
        equipments=[{"oxygen", "telemetry"}],
        nurse_skills=[3],
        nurse_max_load = [10, 12.5, 15],
        rest_rooms=[1],
        occupancy_rates=[0.85],
        fix_nurses=21,
        mode="auto",
    )
    # Example 3 (cf. Brandt et al. 20230 Variation 3)
    gen = generate_instance_set(
        n_eq_instances=10,
        plan_weeks=[2, 4],
        room_sizes=[13, 26],
        room_balances=[[23, 38, 23, 16]],
        equipments=[{"oxygen", "telemetry"}],
        nurse_skills=[3],
        nurse_max_load=[10, 12.5, 15],
        rest_rooms=[1],
        occupancy_rates=[0.85],
        fix_nurses=21,
        mode="auto",
    )
