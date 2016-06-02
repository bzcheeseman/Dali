namespace internal {

    template <int N, int M, typename D>
    struct call_or_recurse;

    template <typename ...Types>
    struct dispatcher {
        template <typename F, typename ...Args>
        static auto impl(F f, const std::tuple<Types...>& params, Args... args) ->
                decltype(call_or_recurse<sizeof...(Args), sizeof...(Types), dispatcher<Types...> >::call(f, params, args...)) {
            return call_or_recurse<sizeof...(Args), sizeof...(Types), dispatcher<Types...> >::call(f, params, args...);
        }
    };

    template <int N, int M, typename D>
    struct call_or_recurse {
        // recurse again
        template <typename F, typename T, typename ...Args>
        static auto call(F f, const T& t, Args... args) -> decltype(D::template impl(f, t, std::get<M-(N+1)>(t), args...)) {
            return D::template impl(f, t, std::get<M-(N+1)>(t), args...);
        }
    };

    template <int N, typename D>
    struct call_or_recurse<N,N,D> {
        // do the call
        template <typename F, typename T, typename ...Args>
        static auto call(F f, const T&, Args... args) -> decltype(f(args...)) {
            return f(args...);
        }
    };
}

template<typename FunctT, typename... Args>
auto unpack_tuple(FunctT f, const std::tuple<Args...>& params) -> decltype(f(std::declval<Args>()...)) {
        // decltype(internal::dispatcher<Args...>::impl(f, params)) {
    return internal::dispatcher<Args...>::impl(f, params);
}
