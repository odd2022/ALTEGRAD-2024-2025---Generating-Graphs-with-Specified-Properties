using JuMP
using CPLEX
using CSV
using DataFrames


function solve(n_nodes::Int, n_edges::Int, average_degree::Float64, triangles_count::Int)
    # Model creation
    model = Model(CPLEX.Optimizer)
    set_optimizer_attribute(model, "CPX_PARAM_TILIM", 20)

    # x_ij = 1 if there is an edge between node i and node j
    @variable(model, x[1:n_nodes, 1:n_nodes], Bin)

    # n_i = 1 if node i is active (if it has at least one incoming edge)    
    @variable(model, n[1:n_nodes], Bin)

    # t_ijk = 1 if ijk is a triangle
    @variable(model, t[1:n_nodes, 1:n_nodes, 1:n_nodes], Bin)

    # Representing the difference between real stats and stats of the generated graph
    @variable(model, nodes_dif >= 0)
    @variable(model, edges_dif >= 0)
    @variable(model, triangles_dif >= 0)
    @variable(model, avg_degree_dif >= 0)

    # Objective function : minimizing the difference of edges, nodes, triangles and avg degree 
    @objective(model, Min, nodes_dif + edges_dif + triangles_dif + avg_degree_dif)

    M = 50

    # Symetry of edges
    for i in 1:n_nodes
        for j in 1:n_nodes
            if i != j
                @constraint(model, x[i, j] == x[j, i])
            end
        end
    end

    # Constraint for the active nodes
    for i in 1:n_nodes
        @constraint(model, n[i] >= sum(x[i, j] for j in 1:n_nodes)/M)
        @constraint(model, n[i] <= sum(x[i, j] for j in 1:n_nodes))
    end

    # Contraint for the existence of a triangle
    for i in 1:n_nodes
        for j in i+1:n_nodes
            for k in j+1:n_nodes
                @constraint(model, t[i, j, k] <= x[i, j])
                @constraint(model, t[i, j, k] <= x[j, k])
                @constraint(model, t[i, j, k] <= x[k, i])
                @constraint(model, t[i, j, k] >= x[i, j] + x[j, k] + x[k, i] - 2)
            end
        end
    end

    # Constraint on the number of nodes
    @constraint(model, sum(n[i] for i in 1:n_nodes) - n_nodes <= nodes_dif)
    @constraint(model, sum(n[i] for i in 1:n_nodes) - n_nodes >= - nodes_dif)


    # Constraint on the number of edges
    @constraint(model, sum(x[i, j] for i in 1:n_nodes, j in i+1:n_nodes) - n_edges <= edges_dif)
    @constraint(model, sum(x[i, j] for i in 1:n_nodes, j in i+1:n_nodes) - n_edges >= - edges_dif)

    # Contraint on the number of triangles
    @constraint(model, sum(t[i, j, k] for i in 1:n_nodes, j in i+1:n_nodes, k in j+1:n_nodes) - triangles_count <= triangles_dif)
    @constraint(model, sum(t[i, j, k] for i in 1:n_nodes, j in i+1:n_nodes, k in j+1:n_nodes) - triangles_count >= - triangles_dif)

    # Contraint on the average node degree
    @constraint(model, sum(x[i, j] for i in 1:n_nodes for j in i+1:n_nodes)/n_nodes - average_degree <= avg_degree_dif)
    @constraint(model, sum(x[i, j] for i in 1:n_nodes for j in i+1:n_nodes)/n_nodes - average_degree >= - avg_degree_dif)

    optimize!(model)

    feasibleSolutionFound = primal_status(model) == MOI.FEASIBLE_POINT
    isOptimal = termination_status(model) == MOI.OPTIMAL
    if feasibleSolutionFound
        edges = JuMP.value.(x)
        nodes_dif = JuMP.value.(nodes_dif)
        edges_dif = JuMP.value.(edges_dif)
        triangles_dif = JuMP.value.(triangles_dif)
        avg_degree_dif = JuMP.value.(avg_degree_dif)
        opt = JuMP.objective_value(model)
    end
    return model, edges, opt
end


function export_solver_results(output_filename)
    # Opening the properties of the test set in a df
    test_real_stats_csv = CSV.File("test_set_real_stats.csv", header=1)
    test_real_stats_df = DataFrame(test_real_stats_csv)

    open(output_filename, "w") do file
        println(file, "graph_id,edge_list")
        for row in eachrow(test_real_stats_df)
            
            graph_id = row.graph
            n_nodes = round(Int, row[2])
            n_edges = round(Int, row[3])
            average_degree = row[4]
            triangles_count = round(Int, row[5])

            model, edges, opt = solve(n_nodes, n_edges, average_degree, triangles_count)

            str_edges = ""
            for i in 1:n_nodes
                for j in i+1:n_nodes
                    if value(edges[i, j]) == 1
                        str_edges *= "($i,$j),"
                    end
                end
            end
            str_edges = chop(str_edges, tail = 1)
            str_edges = "\"" * str_edges * "\""

            println(file, "$graph_id,$str_edges")
        end
    end
end

export_solver_results("output_solver.csv")